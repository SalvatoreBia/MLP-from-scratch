#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <random>
#include <cmath>
#include <algorithm>
#include <cublas_v2.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUT_SIZE 10
#define LR 0.01f
#define BATCH_SIZE 64
#define THREADS_PER_BLOCK 256
#define EPOCHS 10

class Dataset
{
private:
    float *images;
    float *one_hot_enc;
    size_t px_per_img;
    size_t nimages;

private:
    float *d_images;
    float *d_one_hot_enc;

public:
    Dataset(const std::string& filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
            throw std::invalid_argument("wrong filename");

        px_per_img = INPUT_SIZE;
        nimages = 0;

        std::string line;
        std::getline(file, line);

        std::vector<float> temp_images;
        std::vector<float> temp_labels;

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string value;
            
            std::getline(ss, value, ',');
            int label = std::stoi(value);
            for (int i = 0; i < 10; ++i)
                temp_labels.push_back(i == label ? 1.0f : 0.0f);

            while (std::getline(ss, value, ','))
                temp_images.push_back(std::stof(value) / 255.0f);

            nimages++;
        }

        images = new float[temp_images.size()];
        one_hot_enc = new float[temp_labels.size()];

        for (size_t i = 0; i < temp_images.size(); ++i) images[i] = temp_images[i];
        for (size_t i = 0; i < temp_labels.size(); ++i) one_hot_enc[i] = temp_labels[i];
    }

    ~Dataset()
    {
        delete[] images;
        delete[] one_hot_enc;
        free_device();
    }

    void device_alloc()
    {
        d_images = nullptr;
        d_one_hot_enc = nullptr;

        size_t d_images_size = nimages * px_per_img * sizeof(float);
        size_t d_one_hot_enc_size = nimages * 10 * sizeof(float);

        cudaMalloc((void**)&d_images, d_images_size);
        cudaMalloc((void**)&d_one_hot_enc, d_one_hot_enc_size);
        cudaMemcpy(d_images, images, d_images_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_one_hot_enc, one_hot_enc, d_one_hot_enc_size, cudaMemcpyHostToDevice);
    }

    void free_device()
    {
        if (d_images) cudaFree(d_images);
        if (d_one_hot_enc) cudaFree(d_one_hot_enc);
    }

    float* get_d_images() { return d_images; }
    float* get_d_labels() { return d_one_hot_enc; }
    size_t get_size() { return nimages; }
};

//
// ---------------------------------------------------------
//

__global__ void apply_bias_and_ReLU(float *d_z, float *d_b, float *d_a, int tot_elements, int hidden_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tot_elements)
    {
        int col = idx % hidden_size;
        float val = d_z[idx] + d_b[col];
        d_a[idx] = (val < 0.0f) ? 0.0f : val;
    }
}

__global__ void apply_bias_and_softmax(float *d_z, float *d_b, float *d_a, int num_rows, int num_cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows)
    {
        float max_val = -1e20f;
        for (int c = 0; c < num_cols; ++c)
        {
            int idx = row * num_cols + c;
            float val = d_z[idx] + d_b[c];
            if (val > max_val) max_val = val;
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < num_cols; ++c)
        {
            int idx = row * num_cols + c;
            float val = d_z[idx] + d_b[c];
            float e = expf(val - max_val);
            d_a[idx] = e;
            sum_exp += e;
        }

        for (int c = 0; c < num_cols; ++c)
        {
            int idx = row * num_cols + c;
            d_a[idx] /= sum_exp;
        }
    }
}

__global__ void compute_dz2(float *d_a2, float *d_y, float *d_dz2, int tot_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tot_elements)
        d_dz2[idx] = d_a2[idx] - d_y[idx];
}

__global__ void compute_db(float *d_dz, float *d_db, int batch_size, int num_cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_cols)
    {
        float sum = 0.0f;
        for (int r = 0; r < batch_size; ++r)
            sum += d_dz[r * num_cols + col];
        d_db[col] = sum / batch_size;
    }
}

__global__ void compute_dz1(float *d_da1, float *d_z1, float *d_dz1, int tot_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tot_elements)
        d_dz1[idx] = (d_z1[idx] > 0.0f) ? d_da1[idx] : 0.0f;
}

__global__ void update_params(float *param, float *grad, float lr, int tot_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tot_elements)
        param[idx] -= lr * grad[idx];
}

// Kernel per calcolare quante previsioni sono corrette nel batch
__global__ void compute_accuracy(float *d_a2, float *d_y, int *d_correct, int batch_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size)
    {
        int pred_class = 0;
        float max_prob = d_a2[row * 10];
        
        int true_class = 0;
        float max_true = d_y[row * 10];

        for (int c = 1; c < 10; ++c)
        {
            if (d_a2[row * 10 + c] > max_prob)
            {
                max_prob = d_a2[row * 10 + c];
                pred_class = c;
            }
            if (d_y[row * 10 + c] > max_true)
            {
                max_true = d_y[row * 10 + c];
                true_class = c;
            }
        }

        if (pred_class == true_class)
            atomicAdd(d_correct, 1);
    }
}

class MLP
{
private:
    float *d_w1, *d_w2, *d_b1, *d_b2;
    float *d_z1, *d_z2, *d_a1, *d_a2;

    float *d_dz2, *d_dw2, *d_db2;
    float *d_da1, *d_dz1, *d_dw1, *d_db1;

private:
    cublasHandle_t handle;
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const float alpha_m = 1.0f / BATCH_SIZE;

private:
    std::vector<float> init_weights(int n_in, int n_out)
    {
        std::vector<float> weights(n_in * n_out);
        std::random_device rd;
        std::mt19937 gen(rd());
        float std_dev = std::sqrt(2.0f / n_in);
        std::normal_distribution<float> dist(0.0f, std_dev);
        for (auto& w : weights)
            w = dist(gen);
        return weights;
    }

public:
    MLP()
    {
        cublasCreate(&handle);

        cudaMalloc((void**)&d_w1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
        cudaMalloc((void**)&d_w2, OUT_SIZE * HIDDEN_SIZE * sizeof(float));
        cudaMalloc((void**)&d_b1, HIDDEN_SIZE * sizeof(float));
        cudaMalloc((void**)&d_b2, OUT_SIZE * sizeof(float));
        
        cudaMalloc((void**)&d_z1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
        cudaMalloc((void**)&d_z2, BATCH_SIZE * OUT_SIZE * sizeof(float));
        cudaMalloc((void**)&d_a1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
        cudaMalloc((void**)&d_a2, BATCH_SIZE * OUT_SIZE * sizeof(float));
        
        cudaMalloc((void**)&d_dz2, BATCH_SIZE * OUT_SIZE * sizeof(float));
        cudaMalloc((void**)&d_dw2, OUT_SIZE * HIDDEN_SIZE * sizeof(float));
        cudaMalloc((void**)&d_db2, OUT_SIZE * sizeof(float));
        cudaMalloc((void**)&d_da1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
        cudaMalloc((void**)&d_dz1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
        cudaMalloc((void**)&d_dw1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
        cudaMalloc((void**)&d_db1, HIDDEN_SIZE * sizeof(float));

        std::vector<float> w1 = init_weights(INPUT_SIZE, HIDDEN_SIZE);
        std::vector<float> w2 = init_weights(HIDDEN_SIZE, OUT_SIZE);
        
        cudaMemcpy(d_w1, w1.data(), HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w2, w2.data(), HIDDEN_SIZE * OUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_b1, 0, HIDDEN_SIZE * sizeof(float));
        cudaMemset(d_b2, 0, OUT_SIZE * sizeof(float));
    }

    void forward(float *d_X)
    {
        cublasSgemm(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            HIDDEN_SIZE, BATCH_SIZE, INPUT_SIZE,
            &alpha, d_w1, INPUT_SIZE, d_X, INPUT_SIZE,
            &beta, d_z1, HIDDEN_SIZE
        );
    
        int threadsPerBlock = THREADS_PER_BLOCK; 
        int tot_elements_L1 = BATCH_SIZE * HIDDEN_SIZE;
        int blocksPerGrid_L1 = (tot_elements_L1 + threadsPerBlock - 1) / threadsPerBlock;
        apply_bias_and_ReLU<<<blocksPerGrid_L1, threadsPerBlock>>>(d_z1, d_b1, d_a1, tot_elements_L1, HIDDEN_SIZE);
        
        cublasSgemm(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            OUT_SIZE, BATCH_SIZE, HIDDEN_SIZE,
            &alpha, d_w2, HIDDEN_SIZE, d_a1, HIDDEN_SIZE,
            &beta, d_z2, OUT_SIZE
        );

        int blocksPerGrid_L2 = (BATCH_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        apply_bias_and_softmax<<<blocksPerGrid_L2, threadsPerBlock>>>(d_z2, d_b2, d_a2, BATCH_SIZE, OUT_SIZE);
    
        cudaDeviceSynchronize();
    }

    void backward(float *d_X, float *d_Y)
    {
        int threadsPerBlock = THREADS_PER_BLOCK;
        
        int tot_dz2 = BATCH_SIZE * OUT_SIZE;
        int blocks_dz2 = (tot_dz2 + threadsPerBlock - 1) / threadsPerBlock;
        compute_dz2<<<blocks_dz2, threadsPerBlock>>>(d_a2, d_Y, d_dz2, tot_dz2);

        cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_T, 
            HIDDEN_SIZE, OUT_SIZE, BATCH_SIZE, 
            &alpha_m, d_a1, HIDDEN_SIZE, d_dz2, OUT_SIZE, 
            &beta, d_dw2, HIDDEN_SIZE
        );

        int blocks_db2 = (OUT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        compute_db<<<blocks_db2, threadsPerBlock>>>(d_dz2, d_db2, BATCH_SIZE, OUT_SIZE);

        cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            HIDDEN_SIZE, BATCH_SIZE, OUT_SIZE, 
            &alpha, d_w2, HIDDEN_SIZE, d_dz2, OUT_SIZE, 
            &beta, d_da1, HIDDEN_SIZE
        );

        int tot_dz1 = BATCH_SIZE * HIDDEN_SIZE;
        int blocks_dz1 = (tot_dz1 + threadsPerBlock - 1) / threadsPerBlock;
        compute_dz1<<<blocks_dz1, threadsPerBlock>>>(d_da1, d_z1, d_dz1, tot_dz1);

        cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_T, 
            INPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE, 
            &alpha_m, d_X, INPUT_SIZE, d_dz1, HIDDEN_SIZE, 
            &beta, d_dw1, INPUT_SIZE
        );

        int blocks_db1 = (HIDDEN_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        compute_db<<<blocks_db1, threadsPerBlock>>>(d_dz1, d_db1, BATCH_SIZE, HIDDEN_SIZE);

        float lr = LR;
        
        int tot_w2 = OUT_SIZE * HIDDEN_SIZE;
        update_params<<<(tot_w2 + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_w2, d_dw2, lr, tot_w2);
        update_params<<<blocks_db2, threadsPerBlock>>>(d_b2, d_db2, lr, OUT_SIZE);

        int tot_w1 = HIDDEN_SIZE * INPUT_SIZE;
        update_params<<<(tot_w1 + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_w1, d_dw1, lr, tot_w1);
        update_params<<<blocks_db1, threadsPerBlock>>>(d_b1, d_db1, lr, HIDDEN_SIZE);

        cudaDeviceSynchronize();
    }

    int evaluate(float* d_Y)
    {
        int *d_correct;
        int h_correct = 0;
        cudaMalloc((void**)&d_correct, sizeof(int));
        cudaMemcpy(d_correct, &h_correct, sizeof(int), cudaMemcpyHostToDevice);

        int threadsPerBlock = THREADS_PER_BLOCK;
        int blocksPerGrid = (BATCH_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        compute_accuracy<<<blocksPerGrid, threadsPerBlock>>>(d_a2, d_Y, d_correct, BATCH_SIZE);
        
        cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_correct);
        
        return h_correct;
    }

    ~MLP()
    {
        free_device();
        cublasDestroy(handle);
    }

    void free_device()
    {
        if(d_w1) cudaFree(d_w1);
        if(d_w2) cudaFree(d_w2);
        if(d_b1) cudaFree(d_b1);
        if(d_b2) cudaFree(d_b2);
        if(d_z1) cudaFree(d_z1);
        if(d_z2) cudaFree(d_z2);
        if(d_a1) cudaFree(d_a1);
        if(d_a2) cudaFree(d_a2);
        
        if(d_dz2) cudaFree(d_dz2);
        if(d_dw2) cudaFree(d_dw2);
        if(d_db2) cudaFree(d_db2);
        if(d_da1) cudaFree(d_da1);
        if(d_dz1) cudaFree(d_dz1);
        if(d_dw1) cudaFree(d_dw1);
        if(d_db1) cudaFree(d_db1);
    }
};

int main()
{
    try 
    {
        Dataset train("/kaggle/input/competitions/digit-recognizer/train.csv");
        train.device_alloc();

        MLP net;

        float *d_batch_X;
        float *d_batch_Y;
        cudaMalloc((void**)&d_batch_X, BATCH_SIZE * INPUT_SIZE * sizeof(float));
        cudaMalloc((void**)&d_batch_Y, BATCH_SIZE * OUT_SIZE * sizeof(float));

        float *d_full_images = train.get_d_images();
        float *d_full_labels = train.get_d_labels();
        size_t total_images = train.get_size();

        size_t train_size = total_images * 0.8;
        int train_batches = train_size / BATCH_SIZE;
        int val_batches = (total_images - train_size) / BATCH_SIZE;

        for (int epoch = 1; epoch <= EPOCHS; ++epoch)
        {
            int correct_train = 0;
            for (int b = 0; b < train_batches; ++b)
            {
                size_t offset_X = b * BATCH_SIZE * INPUT_SIZE;
                size_t offset_Y = b * BATCH_SIZE * OUT_SIZE;

                cudaMemcpy(d_batch_X, d_full_images + offset_X, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_batch_Y, d_full_labels + offset_Y, BATCH_SIZE * OUT_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);

                net.forward(d_batch_X);
                net.backward(d_batch_X, d_batch_Y);
                
                correct_train += net.evaluate(d_batch_Y);
            }

            int correct_val = 0;
            for (int b = 0; b < val_batches; ++b)
            {
                size_t offset_X = (train_batches + b) * BATCH_SIZE * INPUT_SIZE;
                size_t offset_Y = (train_batches + b) * BATCH_SIZE * OUT_SIZE;

                cudaMemcpy(d_batch_X, d_full_images + offset_X, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_batch_Y, d_full_labels + offset_Y, BATCH_SIZE * OUT_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);

                net.forward(d_batch_X);
                
                correct_val += net.evaluate(d_batch_Y);
            }

            float train_acc = (float)correct_train / (train_batches * BATCH_SIZE) * 100.0f;
            float val_acc = (float)correct_val / (val_batches * BATCH_SIZE) * 100.0f;
            
            std::cout << "Epoch " << epoch << "/" << EPOCHS << " - Train Acc: " << train_acc << "% - Val Acc: " << val_acc << "%\n";
        }

        cudaFree(d_batch_X);
        cudaFree(d_batch_Y);

    } 
    catch (const std::exception& e) 
    {
        std::cerr << e.what() << "\n";
    }

    return 0;
}
