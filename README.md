# dojo-ds 

- [Documentation on readthedocs](http://dojo-ds.readthedocs.io/)  [![Documentation Status](https://readthedocs.org/projects/dojo-ds/badge/?version=latest)](https://dojo-ds.readthedocs.io/en/latest/?badge=latest)




___

# Development Notes

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
- Whenever the repo is updated, the docs are re-built using the settings in `.readthedocs.yml`:
	```yml
	# Build documentation in the docs/ directory with Sphinx
	sphinx:
		configuration: docs/conf.py

	# If using Sphinx, optionally build your docs in additional formats such as PDF
	# formats:
	#    - pdf

	# Optionally declare the Python requirements required to build your docs
	python:
		install:
		- requirements: requirements.txt #requirements-detected.txt
	```
	
- To make the docs locally in the same way as they will be for readthedocs:
	-  Use `make html` command, which will invoke the same `configuration`` file as readthedocs

### Summary of Workflow (WIP):
Normally, this process is done by readthedocs.org as it builds the documentation, but we can be run manually:


- Commands to generate documentation:

```bash
cd docs
make html
```


* Running `make html` command uses Sphinx to generate documentation.
	- Command runs `make.bat`, which references `Makefile` to create the documentation based on the contents of the `docs` folder.
- Sphinx uses `docs/conf.py`'s settings (and 2 functions addd at bottom of file) to run doc creation.
- Sphinx uses  .rst files in the `docs` folder, many of these just reference the all-caps files in the main directory of the repo. 
	- AUTHORS.rst, CONTRIBUTING.rst, HISTORY.rst, README.rst


* Files to Edit:
	- `docs/index.rst`: Primary file for controlling documentation homepage layout
	- `docs/conf.py`:
		- theme, api doc settings, etc.
	- `./AUTHORS/CONTRIBUTING.rst/etc.`
		- Modify to change contents of generated documentation.

- Final Output Folder for build is  `docs/build/html/`, which is set by `docs/Makefile`'s vars

```makefile
SPHINXOPTS    = 
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = dojo_ds
SOURCEDIR     = .
BUILDDIR      = build
```

	



- Critical Files:
	- `docs/conf.py`: Parameters for sphinx docs creation.
	- Used by Sphinx's `make html` command.
		- `Makefile`
		- `make.bat`   



#### Readthedocs

- `.readthedocs.yml`

