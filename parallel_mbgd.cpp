#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <omp.h>
#include <fstream>
#include <sstream>
using namespace std;

int main() {
    cout << "Name : Sreyas Krishnan \n";

    string filename = "dataset25k.csv";
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filename << endl;
        return 1;
    }

    vector<vector<double>> features;
    vector<double> target;
    string line;

    cout << "Loading dataset..." << endl;
    while (getline(file, line)) {
        stringstream ss(line);
        string val;
        vector<double> row;

        while (getline(ss, val, ',')) {
            row.push_back(stod(val));
        }

        if (row.size() < 2) continue; // 
        target.push_back(row.back());
        row.pop_back();
        features.push_back(row);
    }
    file.close();

    int numSamples = features.size();
    int numFeatures = features[0].size();

    cout << "Loaded dataset with " << numSamples 
         << " samples and " << numFeatures << " features.\n";

    vector<double> weights(numFeatures, 0.0), momentumVec(numFeatures, 0.0);
    double bias = 0.0, momentumBias = 0.0;
    double learningRate = 0.05, momentumFactor = 0.9, regLambda = 0.01;
    int maxEpochs = 10, baseBatch = 128;

    vector<int> sampleIndex(numSamples);
    for (int i = 0; i < numSamples; i++) sampleIndex[i] = i;

    double bestLoss = 1e9;
    mt19937 rng(1234);

    double tStart = omp_get_wtime();

    for (int epoch = 0; epoch < maxEpochs; epoch++) {
        shuffle(sampleIndex.begin(), sampleIndex.end(), rng);
        double epochLoss = 0.0;

        int currBatchSize = baseBatch - (epoch / 10) * 16;
        if (currBatchSize < 16) currBatchSize = 16;
        int totalBatches = (numSamples + currBatchSize - 1) / currBatchSize;

        vector<vector<double>> gradW_batch(totalBatches, vector<double>(numFeatures, 0.0));
        vector<double> gradB_batch(totalBatches, 0.0);
        vector<double> loss_batch(totalBatches, 0.0);

        // 🔹 Start gradient computation timer
        double gradStart = omp_get_wtime();

        #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < totalBatches; b++) {
            int startIdx = b * currBatchSize;
            int endIdx = min(startIdx + currBatchSize, numSamples);

            vector<double> localGradW(numFeatures, 0.0);
            double localGradB = 0.0, localLoss = 0.0;

            for (int i = startIdx; i < endIdx; i++) {
                int id = sampleIndex[i];
                double predY = bias;
                for (int j = 0; j < numFeatures; j++)
                    predY += weights[j] * features[id][j];

                double err = predY - target[id];
                for (int j = 0; j < numFeatures; j++)
                    localGradW[j] += err * features[id][j] + regLambda * weights[j];

                localGradB += err;
                localLoss += err * err;
            }
            gradW_batch[b] = localGradW;
            gradB_batch[b] = localGradB;
            loss_batch[b] = localLoss;
        }

        double gradEnd = omp_get_wtime(); // 🔹 End gradient computation timer

        // Aggregate gradients
        vector<double> gradW(numFeatures, 0.0);
        double gradB = 0.0, totalLoss = 0.0;

        for (int b = 0; b < totalBatches; b++) {
            for (int j = 0; j < numFeatures; j++)
                gradW[j] += gradW_batch[b][j];
            gradB += gradB_batch[b];
            totalLoss += loss_batch[b];
        }

        // 🔹 Start parameter update timer
        double updateStart = omp_get_wtime();

        for (int j = 0; j < numFeatures; j++) {
            momentumVec[j] = momentumFactor * momentumVec[j] + learningRate * gradW[j] / numSamples;
            weights[j] -= momentumVec[j];
        }

        momentumBias = momentumFactor * momentumBias + learningRate * gradB / numSamples;
        bias -= momentumBias;

        double updateEnd = omp_get_wtime(); // 🔹 End parameter update timer

        epochLoss = totalLoss / numSamples;
        cout << "Epoch " << epoch + 1 << " | Loss = " << epochLoss
             << " | lr = " << learningRate << endl;
        cout << "Batches: " << totalBatches << endl;
        cout << "Gradient Time: " << gradEnd - gradStart << " sec | "
             << "Update Time: " << updateEnd - updateStart << " sec" << endl;

        if (epochLoss < bestLoss - 1e-6) bestLoss = epochLoss;

        learningRate *= 0.98;
    }

    double tEnd = omp_get_wtime();
    cout << "\nTotal Execution time (seconds): " << tEnd - tStart << endl;

    cout << "\nFinal model: y = " << bias << " + ";
    for (int j = 0; j < numFeatures; j++)
        cout << weights[j] << "*x" << j << ((j < numFeatures - 1) ? " + " : "");
    cout << endl;

    return 0;
}
