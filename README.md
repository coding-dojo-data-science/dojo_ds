# dojo-ds 


- Deployment Installations (mac) [[source](https://packaging.python.org/tutorials/packaging-projects/)]:
    - `python3 -m pip install --upgrade pip`
    - `python3 -m pip install --upgrade build`
    - `python3 -m pip install --upgrade twine`
    - `python3 -m pip install --upgrade bump2version`
    
    
- Deployment workflow:
    1. generate docs with `python docs/conf.py` (optional)
    2. Commit all changes.
    3. Increase version # with bump2version `bump2version patch` or `bump2version minor`
    4. Build distribution archives: `python -m build`
    5. A) Upload to twine: ~~`twine upload dist/*`~~ [only if using general full-account credentials]
    5. B) Upload to twine with an API token:
        - Use the `--repository` flag with the "server" name from $HOME/.pypirc
        - `twine upload --repository cdds dist/*`
    
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
twine upload --repository cdds dist/*
```