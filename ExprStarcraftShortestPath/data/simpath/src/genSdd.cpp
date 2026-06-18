
extern "C" {
	#include <stdio.h>
	#include <stdlib.h>
	#include "sddapi.h"
}

#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <string>
using namespace std;

#define ll long long


ll ONE_TERM = -1, ZERO_TERM = -2;

struct Node {
	ll child[2];
	Node() {
		child[0] = child[1] = ZERO_TERM;
	};
};

vector<SddNode*> f, dp;
vector<Node> vn;
SddManager* manager;
ll cntCall = 0;
SddNode* dfs(ll u, ll lvl, ll maxLvl) {
	cntCall++;
	if (u >= 0 && dp[u] != NULL)
		return dp[u];

	if (u == ONE_TERM) {
		SddNode* ret = sdd_manager_true(manager);
		for (ll i = lvl; i <= maxLvl; i++)
			ret = sdd_conjoin(ret, sdd_negate(f[i], manager), manager);
		return ret;
	}
	if (u == ZERO_TERM) {
		return sdd_manager_false(manager);
	}
	SddNode* ch0 = dfs(vn[u].child[0], lvl+1, maxLvl);
	SddNode* ch1 = dfs(vn[u].child[1], lvl+1, maxLvl);
	
	//cout << u << " " << vn[u].child[0] << " " << vn[u].child[1] << " " << lvl << endl;
	
	SddNode* alpha = sdd_manager_false(manager);
	SddNode* beta;

	beta = sdd_conjoin(f[lvl], ch1, manager);
	alpha = sdd_disjoin(alpha, beta, manager);
	beta = sdd_conjoin(sdd_negate(f[lvl],manager), ch0, manager);
	alpha = sdd_disjoin(alpha, beta, manager);
	
	dp[u] = alpha;
	//sdd_ref(dp[u], manager);
	return dp[u];
}

int main(int argc, char** argv) {

  ll n, m, l;
  scanf("%lld%lld%lld", &n, &m, &l);

  // set up vtree and manager
  SddLiteral var_count = l;

  SddLiteral* var_order = new SddLiteral[l];
  for (ll i = 0; i < l; i++)
  	var_order[i] = i+1;
  
  
  Vtree* vtree = sdd_vtree_new_with_var_order(var_count,var_order,"right");
  manager = sdd_manager_new(vtree);

  //int auto_gc_and_minimize = 0;
  //manager = sdd_manager_create(var_count,auto_gc_and_minimize);
  
  //automatic garbage collection and vtree minimization
  //sdd_manager_auto_gc_and_minimize_on(manager);

  // construct a formula (A^B)v(B^C)v(C^D)
  printf("constructing SDD ... ");

  f = vector<SddNode*>(l+1);
  dp = vector<SddNode*>(m+1, NULL);
  for (ll i = 1; i <= l; i++) {
  	f[i] = sdd_manager_literal(i, manager);
  }

  vn = vector<Node>(m+1);
  for (ll i = 0; i < n; i++) {
  	ll a[3];
	for (ll j = 0; j < 3; j++) {
		string s;
		cin >> s;
		if (s == "T")
			a[j] = ONE_TERM;
		else if (s == "F")
			a[j] = ZERO_TERM;
		else
			a[j] = stoll(s);
	}
	vn[a[0]].child[0] = a[1];
	vn[a[0]].child[1] = a[2];
  }
  
  SddNode* root = dfs(1, 1, l);
  
  sdd_save_as_dot("output/sdd.dot",root);
  sdd_save("output/sdd.sdd",root);
  sdd_vtree_save("output/sdd.vtree",vtree);
  //sdd_vtree_save_as_dot("output/vtree.dot",vtree);
  printf("done\n");


  SddModelCount count = sdd_model_count(root,manager);
  printf("Model count is: %llu\n", count);
  printf("Cntcall is %lld\n", cntCall);
//   int* vars = sdd_variables(root, manager);
//   for (ll i = 1; i <= l; i++)
// 	printf("%u", vars[i]);
//   printf("\n");

  sdd_manager_free(manager);

  return 0;
}
