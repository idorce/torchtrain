rm -rf build dist torchtrain.egg-info
git push
python3 -m pip install --user --upgrade setuptools wheel twine
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/* --verbose
pip install --upgrade torchtrain
pip install --upgrade torchtrain
pip install --upgrade torchtrain