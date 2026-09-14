import urllib.request
import json
import sys

def get_episodes(submission_id):
    url = 'https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes'
    headers = {'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    payload = json.dumps({'submissionId': submission_id}).encode()
    req = urllib.request.Request(url, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            return data.get('episodes', [])
    except Exception as e:
        return []

visited_subs = set()
queue = [55326571, 56093486, 55295016, 55432553]
all_agents_seen = {}

print("Crawling episodes towards the top...")
for step in range(25):
    if not queue:
        break
    queue.sort(key=lambda s: all_agents_seen.get(s, {}).get('score', 0), reverse=True)
    sub_id = queue.pop(0)
    if sub_id in visited_subs:
        continue
    visited_subs.add(sub_id)
    
    eps = get_episodes(sub_id)
    for ep in eps:
        for a in ep.get('agents', []):
            sid = a.get('submissionId')
            upd = a.get('updatedScore')
            init = a.get('initialScore')
            rew = a.get('reward')
            score = upd if upd is not None else init
            if score is not None and sid:
                if sid not in all_agents_seen or score > all_agents_seen[sid]['score']:
                    all_agents_seen[sid] = {'score': score, 'reward': rew, 'ep': ep['id']}
                if sid not in visited_subs and sid not in queue:
                    queue.append(sid)

print("\n--- Top Submissions Discovered ---")
top_subs = sorted(all_agents_seen.items(), key=lambda x: x[1]['score'], reverse=True)
for sid, info in top_subs[:15]:
    print(f"Sub {sid}: Score {info['score']:.1f} | Reward: {info['reward']} | Sample Ep: {info['ep']}")
