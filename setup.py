from setuptools import setup, find_packages

setup(name='fasttenet',
      version='0.0.2',
      description='FastTENET',
      url='http://github.com/cxinsys/fasttenet',
      author='Complex Intelligent Systems Laboratory (CISLAB)',
      author_email='daewon4you@gmail.com',
      license='BSD-3-Clause',
      packages=find_packages(),
      install_requires=['numpy', 'statsmodels', 'networkx', 'tqdm', 'matplotlib', 'omegaconf', 'mate-cxinsys'],
      zip_safe=False,)
