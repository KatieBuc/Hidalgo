nppath=`python3 -c 'import numpy; print(numpy.__path__[0])'`
nppath=$nppath/core/include/numpy
mkdir lib
mkdir lib/python
export PYTHONPATH=$PYTHONPATH:./lib/python/
CFLAGS="-I $nppath" python3 setup.py install --home=./
python3 setup.py build
gs=`find build/ -name "_gibbs*.so" `
cp $gs dimension/
gs=`find lib/python -type d -name dimension`
export PYTHONPATH=$PYTHONPATH:gs
