#include <iostream>
#include <vector>
#include <omp.h>
#include <climits>
using namespace std;

int main() {
    vector<int> arr = {5, 2, 9, 1, 7, 6, 8, 3, 4};

    int min_val = INT_MAX;
    int max_val = INT_MIN;
    int sum = 0;

    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val) reduction(+:sum)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] < min_val) min_val = arr[i];
        if (arr[i] > max_val) max_val = arr[i];
        sum += arr[i];
    }

    double avg = (double)sum / arr.size();

    cout << "Min: " << min_val << endl;
    cout << "Max: " << max_val << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << avg << endl;

    return 0;
}