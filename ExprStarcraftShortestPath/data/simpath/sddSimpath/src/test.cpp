#include <vector>
#include <iostream>
#include <numeric>
using namespace std;

#define ll long long

//number of results should be (2N!)/(N! * 2^N), where conn has size 2N
vector<vector<pair<ll,ll> > > enumeratePats(vector<ll> conn) {

	ll s = conn.size();
	//assert(s%2 == 0);
	
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

int main() {

	vector<ll> conn(6);
	iota(conn.begin(), conn.end(),1);


	vector<vector<pair<ll,ll> > > res = enumeratePats(conn);
	for (ll i = 0; i < res.size(); i++) {
		for (ll j = 0; j < res[i].size(); j++) {
			cout << res[i][j].first << "," << res[i][j].second << " ";
		}
		cout << endl;
	}

}