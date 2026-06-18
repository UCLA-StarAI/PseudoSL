#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <string>
#include <sstream>
using namespace std;

#define ll long long

int main() {
	ll n = 0, m = 0;
	map<ll,ll> m1;
	map<ll,ll> level;
	vector<string> input, output;
	stringstream ss;
	ll cur = 1;
	string linein, lineout;
	
	while (getline(cin, linein)) {
		if (linein == ".")
			break;
		input.push_back(linein);
	}
	reverse(input.begin(), input.end());
	
	for (ll i = 0; i < input.size(); i++) {
		linein = input[i];
		lineout.clear();
		ss.clear();
		ss.str(linein);
		string s[4];
		for (ll j = 0; j < 4; j++) {
			ss >> s[j];
			if (j == 1) {
				m = max(m, stoll(s[j]));
				continue;
			}
			
			if (s[j] != "T" && s[j] != "B") {
				ll a = stoll(s[j]);
				if (m1.find(a) == m1.end())
					m1[a] = cur++;
				s[j] = to_string(m1[a]);
			}
			else if (s[j] == "B")
				s[j] = "F";
			lineout += s[j] + " ";
		}
		level[stoll(s[0])] = stoll(s[1]);
		//output.push_back(lineout);
	}
	
	//fill in obdd from zdd
	for (ll i = 0; i < input.size(); i++) {
		ss.clear();
		ss.str(input[i]);
		string s[4];
		for (ll j = 0; j < 4; j++) {
			ss >> s[j];
		}
		ll par = m1[stoll(s[0])];

		ll ch[4];
		for (ll j = 2; j <= 3; j++) {
			if (s[j] != "T" && s[j] != "B") {
				ch[j] = m1[stoll(s[j])];
				ll diff = level[ch[j]] - level[par];
				ll newch = ch[j];
				if (diff > 1) {
					newch = cur;
					for (ll i = 0; i < diff-2; i++) {
						lineout = to_string(cur) + " " + to_string(cur+1) + " F";
						output.push_back(lineout);
						cur++;
					}
					lineout = to_string(cur) + " " + to_string(ch[j]) + " F";
					output.push_back(lineout);
					cur++;
				}
				s[j] = to_string(newch);
			}
			if (s[j] == "B")
				s[j] = "F";
		}
		lineout = to_string(par) + " " + s[2] + " " + s[3];
		output.push_back(lineout);
	}
	
	
	n = output.size();
	
	cout << n << " " << cur-1 << " " << m << endl;
	for (size_t i = 0; i < output.size(); i++) {
		cout << output[i] << endl;
	}
}