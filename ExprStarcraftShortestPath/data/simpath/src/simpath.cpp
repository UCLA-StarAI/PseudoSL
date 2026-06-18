#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <string>
using namespace std;

#define ls short
#define ll long long

int GRID_SIZE;
int N;
int M;
int NUM_EDGES;

struct Node {
	//level -1 means 1 terminal, -2 means 0 terminal
	Node(ls u) {
		state = vector<ls>(N+1, -1);
		level = u;
		indeg = 0;
		child[0] = child[1] = NULL;
	}
	vector<ls> state;
	ls level;
	Node* child[2];
	ll indeg; // test
	ll id;
};
Node* ONE_TERM;
Node* ZERO_TERM;

inline bool is_one_term(const Node* n) {
	return n->level == -1;
}
inline bool is_zero_term(const Node* n) {
	return n->level == -2;
}

string node_to_string(Node* n) {
	if (is_one_term(n))
		return "T";
	if (is_zero_term(n))
		return "F";
	return to_string(n->id);
}

void change_to_one_term(Node* n) {
	n->level = -1;
	n->id = -1;
	n->child[0] = n->child[1] = NULL;
}
void change_to_zero_term(Node* n) {
	n->level = -2;
	n->id = -2;
	n->child[0] = n->child[1] = NULL;
}

void init_terminals() {
	ONE_TERM = new Node(-1);
	ZERO_TERM = new Node(-2);
	ONE_TERM->id = -1;
	ZERO_TERM->id = -2;
}


vector<ls> last_edge;
vector<pair<ls, ls> > edges;
vector<vector<Node*> > vn;
ll gid = 0;

struct equal_node_construct {
	bool operator()(const Node* n1, const Node* n2) const {
		if (n1->level != n2->level)
			return false;
		if (n1->state.size() != n2->state.size())
			return false;
		for (ls i = 0; i < n1->state.size(); i++)
			if (n1->state[i] != n2->state[i])
				return false;
		return true;
	}
};
struct hash_node_construct {
    size_t operator()(const Node* node ) const {
    	size_t res = node->level;
		ll l = 0, r = min(size_t(node->level+100), node->state.size());
    	for (ll i = l; i < r; i++)
    		res ^= node->state[i] + 0x9e3779b9 + (res << 6) + (res >> 2);
    	return res;
    }
};

struct equal_node_id {
	bool operator()(const Node* n1, const Node* n2) const {
		return n1->id == n2->id;
	}
};

struct equal_node_reduce {
	bool operator()(const Node* n1, const Node* n2) const {
		if (is_one_term(n1) != is_one_term(n2))
			return false;
		if (is_zero_term(n1) != is_zero_term(n2))
			return false;
		return n1->child[0] == n2->child[0] && n1->child[1] == n2->child[1];
	}
};
struct hash_node_reduce {
    size_t operator()(const Node* node ) const {
    	return (size_t)node->child[0] + (size_t)node->child[1];
    }
};

void print_state(const vector<ls> &v) {
	for (ls i = 0; i < v.size(); i++) {
		if (v[i] > 0)
			cout << i << "-" << v[i] << " ";
	}
	cout << endl;
}

void init_edges_bfs() {
	//bfs ordering of edges
	queue<ls> q;
	q.push(1);
	
	ls edge_cnt = 1;
	vector<bool> seen(N+1);
	seen[1] = true;
	while (!q.empty()) {
		ls i = q.front();
		q.pop();
		//right
		ls i1 = i+1;
		if (i1 >= 1 && i1 <= N && i%M) {
			if (!seen[i1]) {
				q.push(i1);
				seen[i1] = true;	
			}
			last_edge[i] = last_edge[i1] = edge_cnt;
			edges[edge_cnt++] = {i, i1};
		}
		//down
		ls i2 = i+M;
		if (i2 >= 1 && i2 <= N) {
			if (!seen[i2]) {
				q.push(i2);
				seen[i2] = true;	
			}
			last_edge[i] = last_edge[i2] = edge_cnt;
			edges[edge_cnt++] = {i, i2};
		}
	}

	cout << edge_cnt << endl;
	for (ls i = 0; i < edge_cnt; i++) {
		cout << edges[i].first << " " << edges[i].second << endl;
	}
}

void init_edges_topdown() {
	//left to right, top to down ordering of edges
	ls edge_cnt = 1;
	vector<bool> seen(N+1);
	seen[1] = true;
	
	for (ls i = 1; i <= N; i++) {	//ith vertex
		//right
		ls i1 = i+1;
		if (i1 >= 1 && i1 <= N && i%M) {
			last_edge[i] = last_edge[i1] = edge_cnt;
			edges[edge_cnt++] = {i, i1};
		}
		
		//down
		ls i2 = i+M;
		if (i2 >= 1 && i2 <= N) {
			last_edge[i] = last_edge[i2] = edge_cnt;
			edges[edge_cnt++] = {i, i2};
		}
	}
/*
	cout << edge_cnt << endl;
	for (ls i = 0; i < edge_cnt; i++) {
		cout << edges[i].first << " " << edges[i].second << endl;
	}
*/
}

/*
	states:
	mate[u] = v	// u and v are ends of an arc
	mate[u] = 0 // middle of an arc
	mate[u] = -1 // not in an arc
	
*/

bool getChild(ls i, ls c, vector<ls> state, vector<ls> &newState, bool &is_loop) {
	ls v1 = edges[i].first, v2 = edges[i].second;
	newState = state;
	if (c) {	//connect edges
		if (state[v1] == 0 || state[v2] == 0)
			return false;
		
		if (state[v1] == v2) {
			is_loop = true;
			state[v1] = state[v2] = 0;
			newState = state;
			return true;	
		}
		
		if (state[v1] == -1) {
			if (state[v2] == -1) {
				state[v1] = v2;
				state[v2] = v1;
			}
			else {
				state[state[v2]] = v1;
				state[v1] = state[v2];
				state[v2] = 0;
			}
		}
		else {
			if (state[v2] == -1) {
				state[state[v1]] = v2;
				state[v2] = state[v1];
				state[v1] = 0;
			}
			else {
				state[state[v1]] = state[v2];
				state[state[v2]] = state[v1];
				state[v1] = 0;
				state[v2] = 0;
			}
		}	
	}
	if (state[v1] != -1 && state[v1] != 0 && i >= last_edge[v1])
		return false;
	if (state[v2] != -1 && state[v2] != 0 && i >= last_edge[v2])
		return false;
	if (state[v1] == -1 && i >= last_edge[v1])
		state[v1] = 0;
	if (state[v2] == -1 && i >= last_edge[v2])
		state[v2] = 0;
	newState = state;
	return true;
}

bool is_solution(vector<ls> v) {
	for (ls i = 0; i < v.size(); i++) {
		if (v[i] > 0)
			return false;
	}	
	return true;
}

Node* construct() {
	ll num_nodes = 0;
	ls num_edges = 2*(N-M);

	// root node connects first and last vertex
	Node* r = new Node(0);
	r->id = gid++;
	r->indeg = 1;	//to prevent zdd-reduction step from deleting root
	r->state[1] = N;
	r->state[N] = 1;
	
	unordered_map<Node*, Node*, hash_node_construct, equal_node_construct> mm;
	vector<queue<Node*> > q(2*(N-M)+2);
	q[0].push(r);
	
	for (ls i = 0; i <= num_edges; i++) {
		//cout << "edge #" << i << endl;
		mm.clear();
		while (!q[i].empty()) {
			Node* cur = q[i].front();
			q[i].pop();
			for (ls c = 0; c < 2; c++) {
				Node* n = new Node(i+1);
				bool is_loop = false;
				bool good = getChild(i+1, c, cur->state, n->state, is_loop);
				if (is_loop) {
					if (is_solution(n->state)) {
						cur->child[c] = ONE_TERM;
						cur->child[c]->indeg++;
					}
					else {
						cur->child[c] = ZERO_TERM;
						cur->child[c]->indeg++;
					}
					delete n;
				}
				else if (good) {
					//print_state(n->state);
					if (mm.find(n) != mm.end()) {
						cur->child[c] = mm[n];
						cur->child[c]->indeg++;
						delete n;
					}
					else {
						num_nodes++;
						vn[i].push_back(n);
						cur->child[c] = n;
						cur->child[c]->indeg++;
						n->id = gid++;
						mm[n] = n;
						q[i+1].push(n);
					}
				}
				else {
					cur->child[c] = ZERO_TERM;
					cur->child[c]->indeg++;
					delete n;
				}
			}
		}
	}

	cout << "Number of nodes is " << num_nodes+1 << endl;
	return r;
}

void reduceToZdd(Node* r) {
	unordered_map<Node*, Node*, hash_node_reduce, equal_node_reduce> mm;
	mm[ONE_TERM] = ONE_TERM;
	mm[ZERO_TERM] = ZERO_TERM;
	ls num_edges = 2*(N-M);
	ll num_del = 0;
	ll num_del2 = 0;
	for (ll i = num_edges; i >= 0; i--) {
		for (ll j = 0; j < vn[i].size(); j++) {
			for (ll c = 0; c <= 1; c++) {
				if (vn[i][j]->child[c] != NULL) {
					Node* ch = vn[i][j]->child[c];

					if (mm.find(ch) != mm.end() && mm[ch] != ch) {
						vn[i][j]->child[c]->indeg--;
						vn[i][j]->child[c] = mm[ch];
						vn[i][j]->child[c]->indeg++;
						num_del++;
					}
				}
			}
			if (vn[i][j]->child[0] != NULL && vn[i][j]->child[1] == ZERO_TERM) {
				vn[i][j]->child[0]->indeg--;
				vn[i][j]->child[1]->indeg--;
				if (is_one_term(vn[i][j]->child[0]))
					change_to_one_term(vn[i][j]);
				if (is_zero_term(vn[i][j]->child[0]))
					change_to_zero_term(vn[i][j]);
				else {
					vn[i][j]->level = vn[i][j]->child[0]->level;
					//order is important
					vn[i][j]->child[1] = vn[i][j]->child[0]->child[1];
					vn[i][j]->child[0] = vn[i][j]->child[0]->child[0];
				}
			}
			else
				mm[vn[i][j]] = vn[i][j];
		}
	}
	cout << "Number of connections deleted in reduction stage is " << num_del << endl;
	
	
	for (ll i = num_edges; i >= 0; i--) {
		for (ll j = 0; j < vn[i].size(); j++) {
			if (vn[i][j]->indeg == 0) {
				delete vn[i][j];
				num_del2++;
			}
		}
	}
	cout << "Number of nodes deleted in reduction stage is " << num_del2 << endl;
}

void reduceToZddSeparateLevel(Node* r) {
	unordered_map<Node*, Node*, hash_node_reduce, equal_node_reduce> mm;
	ls num_edges = 2*(N-M);
	ll num_del = 0;
	ll num_del2 = 0;
	for (ll i = num_edges; i >= 0; i--) {
		mm.clear();
		mm[ONE_TERM] = ONE_TERM;
		mm[ZERO_TERM] = ZERO_TERM;
		for (ll j = 0; j < vn[i].size(); j++) {
			for (ll c = 0; c <= 1; c++) {
				if (vn[i][j]->child[c] != NULL) {
					Node* ch = vn[i][j]->child[c];

					if (mm.find(ch) != mm.end() && mm[ch] != ch) {
						vn[i][j]->child[c]->indeg--;
						vn[i][j]->child[c] = mm[ch];
						vn[i][j]->child[c]->indeg++;
						num_del++;
					}
				}
			}
			if (vn[i][j]->child[0] != NULL && vn[i][j]->child[1] == ZERO_TERM) {
				vn[i][j]->child[0]->indeg--;
				vn[i][j]->child[1]->indeg--;
				if (is_one_term(vn[i][j]->child[0]))
					change_to_one_term(vn[i][j]);
				if (is_zero_term(vn[i][j]->child[0]))
					change_to_zero_term(vn[i][j]);
				else {
					//order is important
					vn[i][j]->child[1] = vn[i][j]->child[0]->child[1];
					vn[i][j]->child[0] = vn[i][j]->child[0]->child[0];
				}
			}
			else
				mm[vn[i][j]] = vn[i][j];
		}
	}
	cout << "Number of connections deleted in reduction stage is " << num_del << endl;
	
	
	for (ll i = num_edges; i >= 0; i--) {
		for (ll j = 0; j < vn[i].size(); j++) {
			if (vn[i][j]->indeg == 0) {
				delete vn[i][j];
				num_del2++;
			}
		}
	}
	cout << "Number of nodes deleted in reduction stage is " << num_del2 << endl;
}

pair<ll,ll> count_zdd(Node* root, unordered_map<Node*, pair<ll,ll>, hash_node_reduce, equal_node_reduce> &mm2, ll MOD) {
	if (is_one_term(root))
		return {1,0};
	if (is_zero_term(root))
		return {0,0};

	if (mm2.find(root) != mm2.end()) {
		return mm2[root];
	}
	
	pair<ll,ll> res0 = count_zdd(root->child[0], mm2, MOD);
	pair<ll,ll> res1 = count_zdd(root->child[1], mm2, MOD);

	ll k = res0.first + res1.first;
	ll of = res0.second + res1.second;

	if (k > MOD) {
		of += k/MOD;
		k %= MOD;	
	}

	mm2[root] = {k,of};
	return {k,of};
}

void interactive(Node* root) {
	string intro = "Interactively search the DAG!";
	cout << intro << endl;
	
	while (!is_one_term(root) && !(is_zero_term(root))) {
		print_state(root->state);
		cout << "in-degree is " << root->indeg << endl;
		cout << "Enter L to search the 0-child, R to search the 1-child!" << endl;
		string ss;
		cin >> ss;
		if (ss[0] == 'L')
			root = root->child[0];
		else
			root = root->child[1];
	}
	if (is_one_term(root))
		cout << "You have reached the one node" << endl;
	else
		cout << "You have reached the zero node" << endl;
	cout << "in-degree is " << root->indeg << endl;
}

void write_helper(Node* r, unordered_set<Node*,hash_node_reduce, equal_node_id> &ms, bool write, ofstream &fout) {
	if (is_one_term(r) || is_zero_term(r))
		return;
	
	if (ms.find(r) != ms.end())
		return;
	
	ms.insert(r);
	
	if (write)
		fout << node_to_string(r) << " " << node_to_string(r->child[0]) << " " << node_to_string(r->child[1]) << endl;
	write_helper(r->child[0], ms, write, fout);
	write_helper(r->child[1], ms, write, fout);
}

void write_zdd_to_file(Node* r, string filename) {
	//freopen(filename.c_str(),"w",stdout);
	ofstream fout(filename);

	unordered_set<Node*,hash_node_reduce, equal_node_id> ms;
	write_helper(r, ms, false, fout);
	
	fout << ms.size() << " " << NUM_EDGES << endl;
	ms.clear();
	write_helper(r, ms, true, fout);
	
	//fclose(stdout);
}

void init_data_structures() {
	N = GRID_SIZE*GRID_SIZE;
	M = GRID_SIZE;
	NUM_EDGES = 2*(N-M);	

	last_edge = vector<ls>(N+1);
	edges = vector<pair<ls, ls> >(2*N);
	vn = vector<vector<Node*> >(2*N);
}

int main(int argc, char *argv[]) {
	
	// parse command line arguments
	if (argc != 2) {
		cout << "usage: " << argv[0] <<" <grid_size>\n";	
		exit(0);
	}
	GRID_SIZE = atoi(argv[1]);
	 
	init_data_structures();
	

	init_edges_topdown();
	init_terminals();
	Node* root = construct();

	//write unreduced ZDD to file
	string fileUnreduced = "output/unreduced.zdd";
	cout << "Writing unreduced ZDD to file " << fileUnreduced << endl;
	write_zdd_to_file(root, fileUnreduced);
	
	
	//write reduced ZDD to file
	reduceToZdd(root);
	string fileReduced = "output/reduced.zdd";
	cout << "Writing reduced ZDD to file " << fileReduced << endl;
	write_zdd_to_file(root, fileReduced);

	unordered_map<Node*, pair<ll,ll>, hash_node_reduce, equal_node_reduce> mm2;
	ll MOD = (ll)(1e17);
	pair<ll,ll> num_paths = count_zdd(root, mm2, MOD);
	
	if (GRID_SIZE < 10)
		cout << "Number of solutions is " << num_paths.first + num_paths.second*MOD << endl;
	else
		cout << "Number of solutions is " << num_paths.first << " + (" << num_paths.second << "*" << MOD << ")" << endl;

	//interactive(root);
}
