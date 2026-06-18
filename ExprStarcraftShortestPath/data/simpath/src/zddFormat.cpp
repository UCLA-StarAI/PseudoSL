#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <string>
using namespace std;

#define ll long long

int main() {
	ll n, m;
	cin >> n >> m;
	
	unordered_map<ll, ll> m1;
	vector<string> output;
	ll cur = 1;
	
	for (ll i = 0; i < n; i++) {
		string line;
		for (ll j = 0; j < 3; j++) {
			string s;
			cin >> s;
			if (s != "T" && s != "F") {
				ll a = stoll(s);
				if (m1.find(a) == m1.end())
					m1[a] = cur++;
				s = to_string(m1[a]);
			}
			line += s + " ";
		}
		output.push_back(line);
	}
	
	
	cout << n << " " << cur-1 << " " << m << endl;
	for (size_t i = 0; i < output.size(); i++) {
		cout << output[i] << endl;
	}
}