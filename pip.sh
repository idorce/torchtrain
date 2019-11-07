rm -rf build dist torchtrain.egg-info
git push
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
sleep 3
pip install --upgrade torchtrain