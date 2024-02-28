Seems like `places.sqlite` is where Firefox stores bookmarks locally. Should be pretty alright to
access this via Python?

Running a simple `fd` command in `~` reveals some results:

```bash
$ fd -HI --extension 'sqlite' | cut -d '/' -f 1-3 | uniq
# .dots/ipython/.ipython
# .mozilla/firefox/ikgk9jo5.default-release
```

Only 2 directories to look though!
```bash
$ fd -HI 'places.sqlite'
# .mozilla/firefox/ikgk9jo5.default-release/places.sqlite
# .mozilla/firefox/ikgk9jo5.default-release/places.sqlite-wal
```

Is this similar on my laptop?
* Yep! But the directory name is different (random 8 chars)

