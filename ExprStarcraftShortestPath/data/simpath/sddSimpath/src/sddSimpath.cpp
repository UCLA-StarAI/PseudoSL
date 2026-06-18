
extern "C" {
	#include <stdio.h>
	#include <stdlib.h>
	#include "sddapi.h"
}

#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <cstdio>
using namespace std;

#define ls short
#define ll long long

const int PI = 198761234;
ll SOURCE_NODE, TARGET_NODE;

string VTREE_FILE, GRAPH_FILE, SDD_FILE;

vector<pair<ls, ls> > edges;
ll num_nodes, num_edges;
ll gid = 1;
SddManager* manager;

struct VtreeData {
	vector<bool> m_edges;
	vector<bool> frontier;
	vector<bool> inner_nodes;
	VtreeData() {
		m_edges.resize(num_edges+1,false);
		frontier.resize(num_nodes+1,false);
		inner_nodes.resize(num_nodes+1,false);	
	}
	void operator|=(const VtreeData &other) {
		for (ll i = 1; i <= num_edges; i++)
			m_edges[i] = (m_edges[i] | other.m_edges[i]);
	}
	void set_frontier() {
		vector<bool> g1(num_nodes+1,false);
		vector<bool> g2(num_nodes+1,false);
		for (ll i = 1; i <= num_edges; i++) {
			if (m_edges[i]) {
				g1[edges[i].first] = g1[edges[i].second] = 1;
				inner_nodes[edges[i].first] = inner_nodes[edges[i].second] = 1;
			}
			else
				g2[edges[i].first] = g2[edges[i].second] = 1;
		}
		for (ll i = 1; i <= num_nodes; i++) {
			frontier[i] = g1[i] && g2[i];
		}
	}
};

struct Node {
	vector<ll> m;	//matching
	ll term;	// 1 is true, 2 is false, -1 is empty
	ll label;	// +/- edge number (i.e. -3 means -C, 2 means +B in final SDD)
	vector<pair<Node*,Node*> > children;
	bool processed;
	Node() {
		m.resize(num_nodes+1);
		for (ll i = 1; i <= num_nodes; i++)
			m[i] = i;
		m[SOURCE_NODE] = -1*TARGET_NODE, m[TARGET_NODE] = -1*SOURCE_NODE;
		term = 0;
		label = 0;
		processed = false;
	}
	Node(const Node *cp) {
		m = cp->m;
		term = cp->term;
		label = cp->label;
		children = cp->children;
		processed = false;
	}
} *ONE_TERM, *ZERO_TERM;

bool is_one_term(const Node* r) {
	return r->term == 1;
}

bool is_zero_term(const Node* r) {
	return r->term == 2;
}

bool is_empty_term(const Node* r) {
	return r->term == -1;
}

struct equal_node {
	bool operator()(const Node* n1, const Node* n2) const {
		if (is_one_term(n1) != is_one_term(n2))
			return false;
		if (is_zero_term(n1) != is_zero_term(n2))
			return false;
		if (is_empty_term(n1) != is_empty_term(n2))
			return false;
		if (n1->label != n2->label)
			return false;
		for (ll i = 1; i <= num_nodes; i++)
			if (n1->m[i] != n2->m[i])
				return false;
		return true;
	}
};
struct hash_node {
    size_t operator()(const Node* node) const {
    	size_t res = 0;
    	for (ll i = 1; i <= num_nodes; i++)
    		res ^= node->m[i] + 0x9e3779b9 + (res << 6) + (res >> 2);
    	return res;
    }
};

void init_terminals() {
	ONE_TERM = new Node();
	ZERO_TERM = new Node();
	ONE_TERM->term = 1;
	ZERO_TERM->term = 2;
	
	ONE_TERM->m[SOURCE_NODE] = ONE_TERM->m[TARGET_NODE] = 0;
}

void init_edges(string graph) {

	freopen(graph.c_str(),"r",stdin);

	string line;
	//SOURCE_NODE = 1, TARGET_NODE = 1;
	while (getline(cin,line)) {
		istringstream iss(line);
		string word;
		iss >> word;

		if (word[0] == 'c') continue;
		if (word == "graph") {
			iss >> num_nodes >> num_edges;
			edges.resize(num_edges+1);
		}
		else {
			ll edge_num = stoi(word);
			ll nodeA, nodeB;
			iss >> nodeA >> nodeB;
			nodeA++, nodeB++; // 1-index nodes
			edges[edge_num] = {nodeA,nodeB};
			//TARGET_NODE = max(TARGET_NODE,max(nodeA,nodeB));
		}
	}
	cout << "Counting simple paths from node " << SOURCE_NODE << " to " << TARGET_NODE << endl;
}

void init_vtree_frontier(Vtree* root) {
	VtreeData* vd = new VtreeData();
	vd->m_edges[sdd_vtree_var(root)] = true;
	
	Vtree* left = sdd_vtree_left(root);
	Vtree* right = sdd_vtree_right(root);
	if (left) {
		init_vtree_frontier(left);
		VtreeData* lvd = (VtreeData*)sdd_vtree_data(left);
		*vd |= *lvd;
	}
	if (right) {
		init_vtree_frontier(right);
		VtreeData* rvd = (VtreeData*)sdd_vtree_data(right);
		*vd |= *rvd;
	}
	vd->set_frontier();	//IMPORTANT! or else internal Node of vtreedata is inconsistent
	sdd_vtree_set_data(vd, root);
	
	// for (ll i = 1; i <= num_nodes; i++) {
	// 	cout << vd->frontier[i];
	// }
	// cout << endl;
}

void printNode(Node* s, bool print=false) {
	if (!print) return;
	if (is_zero_term(s)) {
		cout << "ZERO" << endl;
		return;
	}
	if (is_one_term(s)) {
		cout << "ONE" << endl;
		return;
	}
	
	for (ll i = 1; i <= num_nodes; i++) {
		if (s->m[i] == i)
			cout << '-';
		else if (!s->m[i])
			cout << '*';
		else if (s->m[i] == PI)
			cout << "PI";
		else
			cout << s->m[i];
		cout << '\t';
	}
	cout << endl;
}

bool finished(Node* s) {
	for (ll i = 1; i <= num_nodes; i++)
		if (s->m[i] != 0 && s->m[i] != i)
			return false;
	//printNode(s, true);
	return true;
}

inline ll sgn(ll x) {
	return (x>0)-(x<0);
}

bool isShannon(Vtree* vtree) {
	return sdd_vtree_is_leaf(sdd_vtree_left(vtree)) || sdd_vtree_is_leaf(sdd_vtree_right(vtree));
}

Node* shannonChild(Vtree* vtree, Node* z, bool guess) {
	Node* nz = new Node(z);
	SddLiteral x = sdd_vtree_var(sdd_vtree_left(vtree));
	if (!sdd_vtree_is_leaf(sdd_vtree_left(vtree))) //right child is the leaf
		x = sdd_vtree_var(sdd_vtree_right(vtree));

	ll ua = edges[x].first, ub = edges[x].second;
	
	if (guess) {
		if (!nz->m[ua] || !nz->m[ub]) {
			return ZERO_TERM;
		}
		if (nz->m[ua] == ub && nz->m[ub] == ua) {
			return ZERO_TERM;
		}
		if (nz->m[ua] < 0 && nz->m[ub] < 0 && nz->m[ua] != -1*ua && nz->m[ub] != -1*ub) {	//a and b are reserved
			if (nz->m[ua] == -1*ub && nz->m[ub] == -1*ua) {	//check if reserved for each other
				nz->m[ua] = nz->m[ub] = 0;
				if (finished(nz)) {
					goto label;
					return nz;
				}
			}
 			else
 				return ZERO_TERM;
		}
		else {
			if (nz->m[ua] == -1*ua)	nz->m[ua] = ua;
			if (nz->m[ub] == -1*ub)	nz->m[ub] = ub;
			ll ta = nz->m[ua], tb = nz->m[ub];
			ll sa = sgn(nz->m[ua]), sb = sgn(nz->m[ub]); 
			ll ss = (sa == -1 || sb == -1)? -1 : 1; 
			nz->m[ua] = nz->m[ub] = 0;
			nz->m[sa*ta] = ss*sb*tb;
			nz->m[sb*tb] = ss*sa*ta;
			if (finished(nz)) {
				goto label;
				return nz;
			}
		}
	}

label:
	if (!sdd_vtree_is_leaf(sdd_vtree_left(vtree))) //right child is the leaf
		return nz;
	
	//
	//line 19-21 in paper
	//
	VtreeData* vd = (VtreeData*)sdd_vtree_data(vtree);
	VtreeData* rvd = (VtreeData*)sdd_vtree_data(sdd_vtree_right(vtree));

	for (ll i = 1; i <= num_nodes; i++) {
		//set difference: F(v) \ F(vr)
		if (!(rvd->frontier[i]) && (vd->frontier[i])) {
			if (nz->m[i] != 0 && nz->m[i] != i)
				return ZERO_TERM;
			nz->m[i] = 0;
		}
	}

	if (!sdd_vtree_is_leaf(sdd_vtree_right(vtree)))
		return nz;

	SddLiteral y = sdd_vtree_var(sdd_vtree_right(vtree));
	
	if (finished(nz)) {
		Node* justY = new Node();
		justY->label = -1*y;
		justY->m[SOURCE_NODE] = justY->m[TARGET_NODE] = 0;
		return justY;
	}
	ua = edges[y].first, ub = edges[y].second;

	if (nz->m[ua] == -1*ub && nz->m[ub] == -1*ua) {
		nz->m[ua] = nz->m[ub] = 0;
		if (finished(nz)) {
			Node* justY = new Node();
			justY->label = y;
			justY->m[SOURCE_NODE] = justY->m[TARGET_NODE] = 0;
			return justY;
		}
	}
	return ZERO_TERM;
}

vector<vector<pair<ll, ll> > > enumerateCombination(vector<vector<pair<ll, ll> > > combs) {
	ll mask = 1;
	vector<ll> base(num_nodes+1,0);
	for (ll i = 1; i <= num_nodes; i++) {
		if (combs[i].size()) {
			base[i] = mask;
			mask *= combs[i].size();
		}
	}
	
	vector<vector<pair<ll, ll> > > ret;
	for (ll m = 0; m < mask; m++) {
		vector<pair<ll, ll> > cur(num_nodes+1);
		ll cmask = m;
		for (ll i = num_nodes; i >= 1; i--) {
			if (combs[i].size()) {
				cur[i] = combs[i][cmask/base[i]];
				cmask %= base[i];
			}
		}
		ret.push_back(cur);
	}
	
	return ret;
}

//number of results should be (2N!)/(N! * 2^N), where conn has size 2N
vector<vector<pair<ll,ll> > > enumeratePats(vector<ll> conn) {

	ll s = conn.size();
	assert(s%2 == 0);
	
	vector<vector<pair<ll,ll> > > res;
	if (s == 0) {	//cant make any more pairs
		return vector<vector<pair<ll,ll> > >(1);
	}
	for (ll i = 1; i < s; i++) {
		//connect conn[0] and conn[i], and recursively call the rest
		
		vector<ll> rest;
		for (ll j = 1; j < s; j++)
			if (j != i)
				rest.push_back(conn[j]);
		vector<vector<pair<ll,ll> > > recur = enumeratePats(rest);
		
		for (ll j = 0; j < recur.size(); j++) {
			recur[j].push_back({conn[0], conn[i]});
			res.push_back(recur[j]);
		}
	}
	return res;
}

bool next_combination (ll &mask, ll numElements) {
	if (!mask) return false;

	// details at http://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
	// compute the lexicographically next size-i subset
	ll t = mask | (mask-1);
	mask = (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(mask) + 1));
	return  (mask < (1LL<<numElements));
}


vector<pair<Node*, Node*> > decompChild(Vtree* vtree, Node* z) {
	Vtree* vtl = sdd_vtree_left(vtree);
	Vtree* vtr = sdd_vtree_right(vtree);
	//VtreeData* vd = (VtreeData*)sdd_vtree_data(vtree);
	VtreeData* lvd = (VtreeData*)sdd_vtree_data(vtl);
	VtreeData* rvd = (VtreeData*)sdd_vtree_data(vtr);

	vector<pair<Node*, Node*> > elems;
	vector<bool> common(num_nodes+1,false);
	for (ll i = 1; i <= num_nodes; i++)
		common[i] = lvd->frontier[i] && rvd->frontier[i];

	Node* mp = new Node(z);
	Node* ms = new Node(z);

	for (ll i = 1; i <= num_nodes; i++) {
		if (!(lvd->frontier[i]) && rvd->frontier[i]) {
			mp->m[i] = 0;
		}
		if (!(rvd->inner_nodes[i]) && lvd->frontier[i]) {
			ms->m[i] = 0;
		}
	}

	//
	//	Conditioning on goal node. TODO??
	//
	if (!common[TARGET_NODE]) {
		assert(!(lvd->inner_nodes[TARGET_NODE] && rvd->inner_nodes[TARGET_NODE]));
		if (lvd->inner_nodes[TARGET_NODE]) {
			ms->m[TARGET_NODE] = 0;
		}
		else if (rvd->inner_nodes[TARGET_NODE]) {
			mp->m[TARGET_NODE] = 0;
		}
	}

	if (!common[SOURCE_NODE]) {
		assert(!(lvd->inner_nodes[SOURCE_NODE] && rvd->inner_nodes[SOURCE_NODE]));
		if (lvd->inner_nodes[SOURCE_NODE]) {
			ms->m[SOURCE_NODE] = 0;
		}
		else if (rvd->inner_nodes[SOURCE_NODE]) {
			mp->m[SOURCE_NODE] = 0;
		}
	}



	//line 6 to 14
	
	vector<vector<pair<ll, ll> > > combs(num_nodes+1);
	for (ll i = 1; i <= num_nodes; i++) {
		if (!common[i])	continue;
		if (!mp->m[i]) {
			combs[i] = { {0,0} };
		}
		else if (mp->m[i] == i) {
			combs[i] = { {-1*i,0}, {0,i}, {PI, PI} };
		}
		else if (mp->m[i] == -1*i) {
			combs[i] = { {-1*i,0}, {0,-1*i}, {PI, PI} };
		}
		else {
			combs[i] = { {mp->m[i],0}, {0,ms->m[i]} };
		}
	}
	
	vector<vector<pair<ll, ll> > > combList = enumerateCombination(combs);
	
	for (ll kk = 0; kk < combList.size(); kk++) {
		vector<pair<ll, ll> > vals = combList[kk];
		//line 16
		Node* mpp = new Node(mp);
		Node* mss = new Node(ms);
		
		
		for (ll i = 1; i <= num_nodes; i++) {
			if (!common[i])	continue;	//only need to update affected states

			mpp->m[i] = vals[i].first;
			mss->m[i] = vals[i].second;
		}

		//enumerate connections (line 19)
	
		vector<pair<ll,ll> > connections;
		vector<ll> needConnection;
		vector<pair<ll,ll> > primeConnectPairs;

		for (ll i = 1; i <= num_nodes; i++) {
			ll cl = i, cr = mpp->m[i];

			if (cr == PI)	{
				//common in both frontiers, treat as connection to itself
				connections.push_back({i,i});
			}
			else if (cr > 0 &&
					 cr != cl &&
					 mss->m[cr] == cl) {
				assert(mpp->m[cr] == 0);
				connections.push_back({cl,cr});
			}
			else if (cr < 0) {
				cr *= -1;
				if (mpp->m[cr] == 0) {
					needConnection.push_back(cl);
				}
				else if (cr > cl) {
					primeConnectPairs.push_back({cl,cr});
				}
			}
		}


	    // terminates if there are enough connections or
        // odd number of connections remains.
		if ((needConnection.size() > connections.size()) ||
			((connections.size() - needConnection.size()) % 2 == 1)) {
			continue;
		}
		
		ll maxPrimeConnectPairs = min((connections.size()-needConnection.size())/2, primeConnectPairs.size());

		for(ll i = 0; i <= maxPrimeConnectPairs; i++) {
			//initialize a subset of size i
			ll mask = (1LL<<i)-1;
			do {
				vector<ll> pairsCombination;
				for (ll j = 0; j < primeConnectPairs.size(); j++) {
					if (mask & (1LL<<j))
						pairsCombination.push_back(j);
				}

				// need 1*needConnection + 2*primePairs number connections
				ll needConnectCombsSize = needConnection.size() + pairsCombination.size()*2;
				vector<ll> needConnectCombs(needConnectCombsSize);

				ll c_mask = (1LL<<needConnectCombsSize)-1;
				do {
					needConnectCombs.clear();
					for (ll j = 0; j < connections.size(); j++) {
						if (c_mask & (1LL<<j)) {
							needConnectCombs.push_back(j);
						}
					}

					vector<ll> perms(needConnectCombsSize);
					iota(perms.begin(), perms.end(), 0);
					assert(perms.size() == needConnectCombs.size());

					vector<ll> remainConnections; // conections not used for needConnection and primePairs
					{
						ll combIdx = 0;
						for (ll j = 0; j < connections.size(); j++) {
							if (combIdx < needConnectCombsSize && needConnectCombs[combIdx] == j) {
								combIdx += 1;
							}
							else {
								remainConnections.push_back(j);
							}
						}
					}

					assert(remainConnections.size() == (connections.size() - needConnectCombs.size()));

					do {
						//lines 24-26
						Node* mppp = new Node(mpp);
						Node* msss = new Node(mss);

						//update mppp and msss by reflecting connections of needConnect
						for (ll j = 0; j < needConnection.size(); j++) {
							ll needConnectItem = needConnection[j];
							pair<ll,ll> gatePair = connections[needConnectCombs[perms[j]]];
							ll tmp = mppp->m[needConnectItem];
							mppp->m[needConnectItem] = -1*gatePair.first;
							mppp->m[gatePair.first] = -1*needConnectItem;
							msss->m[gatePair.second] = tmp;
							msss->m[-1*tmp] = -1*gatePair.second;
						}


						//update mppp and msss by reflecting connections of primeConnectPairs
						for (ll j = 0; j < pairsCombination.size(); j++) {
							pair<ll,ll> primePair = primeConnectPairs[pairsCombination[j]];
							ll l  = 2*j + needConnection.size();
							pair<ll,ll> pairFirstConnection = connections[needConnectCombs[perms[l]]];
							pair<ll,ll> pairSecondConnection = connections[needConnectCombs[perms[l+1]]];

							mppp->m[primePair.first] = -1*pairFirstConnection.first;
							mppp->m[pairFirstConnection.first] = -1*primePair.first;

							mppp->m[primePair.second] = -1*pairSecondConnection.first;
							mppp->m[pairSecondConnection.first] = -1*primePair.second;

							msss->m[pairFirstConnection.second] = -1*pairSecondConnection.second;
							msss->m[pairSecondConnection.second] = -1*pairFirstConnection.second;
						}



						//update mppp and msss by reflecting connections of two connection pairs
						vector<ll> v_pairs(connections.size()-needConnectCombs.size());
						iota(v_pairs.begin(),v_pairs.end(),0);
						vector<vector<pair<ll,ll> > > allPairsPartPatterns = enumeratePats(v_pairs);
						
						// genAllPairPartitions(n) is the set of all possible partitions of pairs.
                        // If n = 4, then genAllPairPartitions(4) returns {((1, 2), (3, 4)), ((1, 3), (2, 4)),  ((1, 4), (2, 3))}.
                        // allPairPartPatterns are used to determine the possible pairs of remaining connections.

						if (!allPairsPartPatterns.empty()) {
							for (ll j = 0; j < allPairsPartPatterns.size(); j++) {
								vector<pair<ll,ll> > partition = allPairsPartPatterns[j];

								Node* mp4 = new Node(mppp);
								Node* ms4 = new Node(msss);

								for (ll k = 0; k < partition.size(); k++) {
									pair<ll,ll> pp = partition[k];
									ll parity = 1;
									if (pp.first > pp.second)
										parity = -1;
									pair<ll,ll> pairA = connections[remainConnections[pp.first]];
									pair<ll,ll> pairB = connections[remainConnections[pp.second]];

									mp4->m[pairA.first] = -1*parity*pairB.first;
									mp4->m[pairB.first] = -1*parity*pairA.first;

									ms4->m[pairA.second] = 1*parity*pairB.second;
									ms4->m[pairB.second] = 1*parity*pairA.second;
								}
								elems.push_back({mp4,ms4});
							}
						}
						else {
							elems.push_back({mppp,msss});
						}

					} while (next_permutation(perms.begin(), perms.end()));

				} while (next_combination(c_mask,connections.size()));

			} while (next_combination(mask,primeConnectPairs.size()));
		}
	}
	return elems;
}

void construct(Vtree* vtree, map<Vtree*, vector<Node*> > &Z) {

	unordered_map<Node*, Node*, hash_node, equal_node> mml, mmr;

	Vtree* vtl = sdd_vtree_left(vtree);
	Vtree* vtr = sdd_vtree_right(vtree);
	vector<Node*> vz = Z[vtree];	//check initialization
	if (isShannon(vtree)) {
		for (ll i = 0; i < vz.size(); i++) {
			Node* z = vz[i];
			if (z->processed) continue;
			
			Node* mf = shannonChild(vtree, z, false);
			Node* mt = shannonChild(vtree, z, true);

			SddLiteral x = sdd_vtree_var(vtl);

			if (!sdd_vtree_is_leaf(vtl)) //right child is the leaf
				x = sdd_vtree_var(vtr);
			//cout << "Trying: " << edges[x].first << " " << edges[x].second << " edge #" << x << endl;
			//printNode(z,true);
			//cout << "left: \t";
			//printNode(mf,true);
			//cout << "right: \t";
			//printNode(mt,true);
			
			if (!is_zero_term(mf)) {
				Node* justX = new Node();
				justX->label = -1*x;
				if (!sdd_vtree_is_leaf(vtl))
					z->children.push_back({mf,justX});
				else
					z->children.push_back({justX, mf});
			}
			if (!is_zero_term(mt)) {
				Node* justX = new Node();
				justX->label = x;
				if (!sdd_vtree_is_leaf(vtl))
					z->children.push_back({mt,justX});
				else
					z->children.push_back({justX, mt});
			}
			
			for (ll i = 0; i < z->children.size(); i++) {
				if (mml.find(z->children[i].first) == mml.end())
					mml[z->children[i].first] = z->children[i].first;
				if (mmr.find(z->children[i].second) == mmr.end())
					mmr[z->children[i].second] = z->children[i].second;	
				z->children[i].first = mml[z->children[i].first];
				z->children[i].second = mmr[z->children[i].second];
				Z[vtl].push_back(z->children[i].first);
				Z[vtr].push_back(z->children[i].second);
			}
			
			z->processed = true;
		}
		assert(sdd_vtree_is_leaf(vtl) || sdd_vtree_is_leaf(vtr));
		if (!sdd_vtree_is_leaf(vtl))
			construct(vtl, Z);
		if (!sdd_vtree_is_leaf(vtr))
			construct(vtr, Z);
	}
	else {
		// cout << "decomp child begin" << endl;
		// cout << "-----------------------------------------------------------------------------------" << endl;
		for (ll i = 0; i < vz.size(); i++) {
			Node* z = vz[i];
			if (z->processed) continue;

			z->children = decompChild(vtree, z);	//optimize later using uniqueness
			for (ll i = 0; i < z->children.size(); i++) {
				if (mml.find(z->children[i].first) == mml.end())
					mml[z->children[i].first] = z->children[i].first;
				if (mmr.find(z->children[i].second) == mmr.end())
					mmr[z->children[i].second] = z->children[i].second;
				z->children[i].first = mml[z->children[i].first];
				z->children[i].second = mmr[z->children[i].second];
				Z[vtl].push_back(z->children[i].first);
				Z[vtr].push_back(z->children[i].second);
				
				
				// cout << "children " << i << endl;
				// printNode(z->children[i].first, true);
				// printNode(z->children[i].second, true);
			}
			
			z->processed = true;
		}
		construct(vtl, Z);
		construct(vtr, Z);
	}
}

Node* rootNode() {
	Node* ret = new Node();
	return ret;
}

ll count_sdd(Vtree* vtree, Node* r) {
	if (is_zero_term(r))
		return 0;
	if (is_one_term(r))
		return 1;
	if (is_empty_term(r))
		return 0;
	if (r->label != 0)
		return 1;

	ll count = 0;
	
	Vtree* lt = sdd_vtree_left(vtree);
	Vtree* rt = sdd_vtree_right(vtree);
	
	for (ll i = 0; i < r->children.size(); i++) {
		Node* left = r->children[i].first;
		Node* right = r->children[i].second;
		
		ll lc = count_sdd(lt, left);
		ll rc = count_sdd(rt, right);
		
		count += lc*rc;
	}
	return count;
}

SddNode* dfs(Vtree* vtree, Node* r) {
	if (is_zero_term(r))
		return sdd_manager_false(manager);
	if (is_one_term(r))
		return sdd_manager_true(manager);
	if (is_empty_term(r))
		return sdd_manager_true(manager);
	if (r->label != 0) {
		if (r->label > 0)
			return sdd_manager_literal(r->label, manager);
		else
			return sdd_negate(sdd_manager_literal(-1*r->label, manager), manager);
	}

	Vtree* lt = sdd_vtree_left(vtree);
	Vtree* rt = sdd_vtree_right(vtree);
	
	SddNode* alpha = sdd_manager_false(manager);
	SddNode* beta;
	
	for (ll i = 0; i < r->children.size(); i++) {
		Node* left = r->children[i].first;
		Node* right = r->children[i].second;
		
		SddNode* sl = dfs(lt, left);
		SddNode* sr = dfs(rt, right);
		
		beta = sdd_conjoin(sl, sr, manager);
		alpha = sdd_disjoin(alpha, beta, manager);
	}
	return alpha;
}

void print_vtree(Vtree* vtree) {
	if (!vtree) return;
	cout << sdd_vtree_var(vtree) << endl;
	
	Vtree* vtl = sdd_vtree_left(vtree);
	Vtree* vtr = sdd_vtree_right(vtree);
	
	if (vtl) {
		cout << "Left is: " << sdd_vtree_var(vtl) << endl;
	}
	if (vtr) {
		cout << "Right is: " << sdd_vtree_var(vtr) << endl;
	}
	print_vtree(vtl);
	print_vtree(vtr);
}

int main(int argc, char** argv) {
// will fail on empty vtree, easy to fix if needed

	// parse command line arguments
	if (argc != 6) {
		cout << "usage: " << argv[0] <<" <graph> <vtree> <source> <destination> <output_sdd>\n";	
		exit(0);
	}
	GRAPH_FILE = argv[1];
	VTREE_FILE = argv[2];
    SOURCE_NODE = atoi(argv[3]);
    TARGET_NODE = atoi(argv[4]);
    SDD_FILE = argv[5];

	init_edges(GRAPH_FILE);	
  	init_terminals();

	Vtree* vtree = sdd_vtree_read(VTREE_FILE.c_str());
	manager = sdd_manager_new(vtree);
		
	init_vtree_frontier(vtree);
	
	map<Vtree*, vector<Node*> > Z;
	Z[vtree] = vector<Node*>(1,rootNode());
	construct(vtree, Z);

	ll ans = count_sdd(vtree, Z[vtree][0]);
	cout << "Count is:     " << ans << endl;
	
	SddNode* sddRoot = dfs(vtree, Z[vtree][0]);
	cout << "Sdd Count is: " << sdd_model_count(sddRoot, manager) << endl;
	// sdd_save_as_dot("output/sdd.dot",sddRoot);
	
	sdd_save(SDD_FILE.c_str(),sddRoot);
 //  	sdd_vtree_save("output/sdd.vtree",vtree);

	sdd_manager_free(manager);

  	return 0;
}
