#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

void parallel_bubble_sort(vector<int>& arr) {
    int n = arr.size();

    for (int phase = 0; phase < n; phase++) {

        if (phase % 2 == 0) {
            #pragma omp parallel for
            for (int i = 0; i < n - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    swap(arr[i], arr[i + 1]);
                }
            }
        } else {
            #pragma omp parallel for
            for (int i = 1; i < n - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    swap(arr[i], arr[i + 1]);
                }
            }
        }
    }
}

int main() {
    vector<int> arr = {5, 2, 9, 1, 7, 6, 8, 3, 4};

    double start, end;

    start = omp_get_wtime();
    parallel_bubble_sort(arr);
    end = omp_get_wtime();

    cout << "Parallel Bubble Sort Time: " << end - start << endl;

    for (int x : arr) cout << x << " ";

    return 0;
}
