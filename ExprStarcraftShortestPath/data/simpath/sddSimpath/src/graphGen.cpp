#include <iostream>

using namespace std;

#define ll long long

const ll GRID_SIZE = 30;
const ll M = GRID_SIZE;
const ll N = GRID_SIZE*GRID_SIZE;
pair<ll,ll> edges[2*N];
ll edge_cnt = 1;

void init_edges_topdown() {
	//left to right, top to down ordering of edges
	
	for (ll i = 1; i <= N; i++) {	//ith vertex
		//right
		ll i1 = i+1;
		if (i1 >= 1 && i1 <= N && i%M) {
			edges[edge_cnt++] = {i, i1};
		}
		
		//down
		ll i2 = i+M;
		if (i2 >= 1 && i2 <= N) {
			edges[edge_cnt++] = {i, i2};
		}
	}
}

int main() {

	init_edges_topdown();

	//header
	cout << "graph " << N+1 << " " << edge_cnt-1 << endl;

	for (ll i = 1; i < edge_cnt; i++) {
		cout << i << " " << edges[i].first << " " << edges[i].second << endl;
	}


}
