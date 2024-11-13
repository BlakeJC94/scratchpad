
export SPOTIPY_CLIENT_ID='35466e0bcb7241cf9298838fbdff59f7'
export SPOTIPY_CLIENT_SECRET='fc4f35629c314dc4b6a729bb7ec48dba'
export SPOTIPY_REDIRECT_URI='http://127.0.0.1:9090'

# %%
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# %%
scope = "user-library-read"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

results = sp.current_user_saved_tracks()
for idx, item in enumerate(results['items']):
    track = item['track']
    print(idx, track['artists'][0]['name'], " – ", track['name'])

# %%
from collections import defaultdict

scope = "user-top-read"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
results = sp.current_user_top_artists()

genres = defaultdict(lambda: 0)
for idx, item in enumerate(results['items']):
    print(idx, " - ", item['name'])
    for genre in item['genres']:
        genres[genre] += 1

for genre, count in sorted(list(genres.items()), key=lambda x: -x[1]):
    print(count, genre)
