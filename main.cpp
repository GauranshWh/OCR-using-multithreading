#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <iostream>
#include <filesystem>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <fstream>
#include <algorithm> // For std::transform

// Namespace for filesystem
namespace fs = std::filesystem;

// Thread Pool Implementation
class ThreadPool {
public:
    ThreadPool(size_t numThreads);
    ~ThreadPool();
    void enqueueTask(std::function<void()> task);

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;

    void workerThread();
};

ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back([this]() { workerThread(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) {
        worker.join();
    }
}

void ThreadPool::enqueueTask(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        tasks.push(std::move(task));
    }
    condition.notify_one();
}

void ThreadPool::workerThread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            condition.wait(lock, [this] { return stop || !tasks.empty(); });
            if (stop && tasks.empty()) return;
            task = std::move(tasks.front());
            tasks.pop();
        }
        task();
    }
}

// OCR Processing Function
void processImage(const std::string& imagePath, const std::string& outputPath, std::mutex& outputMutex) {
    tesseract::TessBaseAPI tess;
    if (tess.Init(NULL, "eng")) { // Initialize Tesseract with English
        std::cerr << "Could not initialize tesseract for " << imagePath << std::endl;
        return;
    }

    Pix* image = pixRead(imagePath.c_str());
    if (!image) {
        std::cerr << "Could not read image: " << imagePath << std::endl;
        return;
    }

    tess.SetImage(image);
    std::string text = tess.GetUTF8Text();

    // Write extracted text to the output file
    {
        std::lock_guard<std::mutex> lock(outputMutex);
        std::ofstream outputFile(outputPath, std::ios::app); // Append mode
        if (outputFile) {
            outputFile << "File: " << imagePath << "\n";
            outputFile << text << "\n";
            outputFile << "--------------------\n";
        } else {
            std::cerr << "Error writing to output file for " << imagePath << std::endl;
        }
    }

    pixDestroy(&image);
    tess.End();
}

std::vector<std::string> getImagesFromFolder(const std::string& folderPath) {
    std::vector<std::string> imageFiles;
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            // Get the file extension
            std::string extension = filePath.substr(filePath.find_last_of('.') + 1);

            // Convert the extension to lowercase for case-insensitive comparison
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

            // Filter for common image file extensions
            if (extension == "png" || extension == "jpg" || extension == "jpeg") {
                imageFiles.push_back(filePath);
            }
        }
    }
    return imageFiles;
}

int main() {
    // Specify the folder containing images
    const std::string folderPath = "/home/gauransh-bhatnagar/Downloads/archive/images";  // Update with correct absolute path

    const std::string outputPath = "ocr_output.txt";

    // Get list of image files in the folder
    std::vector<std::string> imageFiles = getImagesFromFolder(folderPath);
    if (imageFiles.empty()) {
        std::cerr << "No image files found in the folder: " << folderPath << std::endl;
        return 1;
    }

    // Clear output file
    std::ofstream clearFile(outputPath, std::ios::trunc);
    if (!clearFile) {
        std::cerr << "Error creating output file: " << outputPath << std::endl;
        return 1;
    }

    // Mutex for synchronizing output file access
    std::mutex outputMutex;

    // Create a thread pool
    size_t numThreads = std::thread::hardware_concurrency(); // Use the number of available cores
    ThreadPool pool(numThreads);

    // Enqueue tasks for OCR processing
    for (const auto& imagePath : imageFiles) {
        pool.enqueueTask([&outputMutex, &outputPath, imagePath]() {
            processImage(imagePath, outputPath, outputMutex);
        });
    }

    std::cout << "Processing " << imageFiles.size() << " images with " << numThreads
              << " threads. Results will be saved to " << outputPath << std::endl;

    return 0;
}

