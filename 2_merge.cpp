#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

/* Merge Function */
void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];

    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];

    while (j < n2)
        arr[k++] = R[j++];
}

/* Sequential Merge Sort */
void sequential_merge_sort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        sequential_merge_sort(arr, l, m);
        sequential_merge_sort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

/* Parallel Merge Sort */
void parallel_merge_sort_util(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;

        #pragma omp task shared(arr)
        parallel_merge_sort_util(arr, l, m);

        #pragma omp task shared(arr)
        parallel_merge_sort_util(arr, m + 1, r);

        #pragma omp taskwait
        merge(arr, l, m, r);
    }
}

void parallel_merge_sort(vector<int>& arr) {
    #pragma omp parallel
    {
        #pragma omp single
        parallel_merge_sort_util(arr, 0, arr.size() - 1);
    }
}

/* Main Function */
int main() {
    vector<int> arr = {5, 2, 9, 1, 7, 6, 8, 3, 4};
    vector<int> arr_seq = arr;
    vector<int> arr_par = arr;

    double start, end;

    // Sequential Merge Sort
    start = omp_get_wtime();
    sequential_merge_sort(arr_seq, 0, arr_seq.size() - 1);
    end = omp_get_wtime();
    cout << "Sequential Merge Sort Time: " << end - start << endl;

    // ✅ Print Sequential Sorted Array
    for (int x : arr_seq) cout << x << " ";
    cout << endl;

    // Parallel Merge Sort
    start = omp_get_wtime();
    parallel_merge_sort(arr_par);
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << end - start << endl;

    // ✅ Print Parallel Sorted Array
    for (int x : arr_par) cout << x << " ";
    cout << endl;

    return 0;
}
