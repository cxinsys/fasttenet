from setuptools import setup, find_packages

setup(name='fasttenet',
      version='1.0.5',
      description='FastTENET',
      url='http://github.com/cxinsys/fasttenet',
      author='Complex Intelligent Systems Laboratory (CISLAB)',
      author_email='daewon4you@gmail.com',
      license='BSD-3-Clause',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=['numpy', 'statsmodels', 'networkx', 'tqdm', 'matplotlib', 'omegaconf', 'mate-cxinsys'],
      zip_safe=False,)
