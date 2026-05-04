#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

const int MAXN = 100000;
vector<int> adj[MAXN + 5];
bool visited[MAXN + 5];

void dfs(int node) {
    visited[node] = true;
    #pragma omp critical

    cout << node << " ";

    #pragma omp parallel for
    for (int i = 0; i < adj[node].size(); i++) {
        int next = adj[node][i];

        bool do_visit = false;

        #pragma omp critical
        {
            if (!visited[next]) {
                visited[next] = true;
                do_visit = true;
            }
        }

        if (do_visit) {
            dfs(next);
        }
    }
}

int main() {
    int n, m;
    cin >> n >> m;

    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int start;
    cin >> start;

    dfs(start);

    return 0;
}


// INPUT:
// 5 4
// 1 2
// 1 3
// 2 4
// 3 5
// 1
