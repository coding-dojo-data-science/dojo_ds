# dojo-ds 


- Deployment Installations (mac) [[source](https://packaging.python.org/tutorials/packaging-projects/)]:
    - `python3 -m pip install --upgrade pip`
    - `python3 -m pip install --upgrade build`
    - `python3 -m pip install --upgrade twine`
    - `python3 -m pip install --upgrade bump2version`
    > Note: try removing the `python3 -m` part of the commands above if you run into issues.
    
- Deployment workflow:
    1. generate docs (locally):
		- with `python docs/conf.py` (optional)
		- Change dir to "docs" folder and run `make html`
		```bash
		cd docs/
		make html
		```
		- Then can open docs/build/index.html in local browser.

    2. Commit all changes.
    3. Increase version # with bump2version `bump2version patch` or `bump2version minor`
    4. Build distribution archives: `python -m build`
    5. A) Upload to twine: `twine upload dist/*` [only if using general full-account credentials]
    5. B) Upload to twine with an API token: [If using a token]
        - Use the `--repository` flag with the "server" name from $HOME/.pypirc
        - `twine upload --repository dojo_ds dist/*`
    
- [11/28/22 Update] Using project-based API token for upload
    - Follow the following guides to set up your own "$HOME/.pypirc" file with the API token: https://pypi.org/help/#apitoken
    - For additional info on the twine upload commands with project APIs: https://kynan.github.io/blog/2020/05/23/how-to-upload-your-package-to-the-python-package-index-pypi-test-server 
    
```bash
pip install --upgrade bump2version
pip install --upgrade pip
pip install --upgrade build
pip install --upgrade twine
```
```
bump2version patch #or minor/major
```
### After install and bump2version, can run this block:
```bash
python -m build 
twine upload dist/*
```
<!-- X twine upload --repository dojo_ds dist/* -->

## Updating the Documentation

- The readthedocs site for this package is created using sphinx and autodoc.
	
### Summary of Workflow (WIP):
Normally, this process is done by readthedocs.org as it builds the documentation, but we can 

> Commands to generate documentation:


```bash
cd docs
make html
```


1. Running `make html` command uses Sphinx to generate documentation.
	- Command runs `make.bat`, which references `Makefile`
	- Sphinx uses `conf.py`'s settings (and functions at bottom of file) to run autodocumentation creation.
	- This creates several .rst files in the `docs` folder. 
		- e.g. dojo_ds.rst, 
		-  some of these are created based on .rst files in the main directory of this repo. 
			- AUTHORS.rst, CONTRIBUTING.rst, HISTORY.rst, README.rst
	- Which are then converted in html files (Final html files are in `docs/build/html/`)

	



- Critical Files:
	- `docs/conf.py`: Parameters for sphinx docs creation.
	- Used by Sphinx's `make html` command.
		- `Makefile`
		- `make.bat`   



#### Readthedocs

- `.readthedocs.yml`

