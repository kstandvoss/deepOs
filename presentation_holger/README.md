On IKW computers I had to compile hdf5, sqlite3 and python3 as follows:

```
mkdir -p /some_directory/installers/hdf5
wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.16.tar
tar -xf hdf5-1.8.16.tar
cd hdf5-1.8.16
./configure --prefix=/some_directory/local --enable-shared
make
make install

mkdir -p /some_directory/installers/sqlite3
wget https://www.sqlite.org/2016/sqlite-autoconf-3120200.tar.gz
tar -xf sqlite-autoconf-3120200.tar.gz
cd sqlite-autoconf-3120200/
./configure --prefix=/some_directory/local
make
make install

mkdir -p /some_directory/installers/python3
cd /some_directory/installers/python3
wget https://www.python.org/ftp/python/3.5.1/Python-3.5.1.tar.xz
tar -xf Python-3.5.1.tar.xz
cd Python-3.5.1
export LDFLAGS="-L/some_directory/local/lib"
export CFLAGS="-L/some_directory/local/include"
./configure --with-ensurepip=install --prefix=/some_directory/local --enable-shared
make
make install
```

Make sure that your PATHs are set accordingly:
```
export PATH=$PATH:/some_directory/local/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/some_directory/local/lib
```

To create a new python virtual env:
```
cd /some_directory/deepOs
pyvenv venv
```

To activate the virtual environment:
```
source venv/bin/activate
```

Now install all required python libraries:
```
pip install -U pip
pip install -U setuptools
pip install -U jupyter
pip install -U cython
pip install -U numpy
pip install -U scipy
pip install -U matplotlib
HDF5_DIR=/some_directory/local pip install -U h5py
pip install -U scikit-learn
pip install -U pydot-ng
pip install -U sympy
pip install -U nose
pip install -U pillow
pip install -U pandas
pip install -U patsy
pip install -U statsmodels
pip install https://github.com/ipython-contrib/IPython-notebook-extensions/archive/master.zip

# install development version of theano:
git clone git://github.com/Theano/Theano.git
cd Theano
python setup.py develop
cd ..

# install development version of keras:
git clone https://github.com/fchollet/keras.git
cd keras
python setup.py develop
cd ..
```

Now start Jupyter Notebook with:
```
jupyter notebook
```

I would enable a few jupyter notebook extensions (i.e. Codefolding) by going to [http://localhost:8888/nbextensions](http://localhost:8888/nbextensions).