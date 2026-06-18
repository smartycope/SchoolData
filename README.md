# SchoolData
Filtering charter school bonds

To test stlite locally:
```bash
google-chrome-insecure /home/zeke/hello/SchoolData/index.html
```

* `streamlit_main.py` is the main entrypoint.
* `notes.md` has some random notes about how charter schools and bonds work

## TODO:
* Limit the number of rows that can be seen at once (add pagination, probably?)
* The desktop scraper needs re-implemented in streamlit on Marvin
    * it uses selenium, can that be run on aarch64?
* Move all the data to Marvin and out of the repo
* The thing *wink*
* More instructions?
* Remove encryption, clean up code, split into multiple files for organization
    * Just use secrets for password authentication instead of a hash
* Move to a virtual env on Marvin
