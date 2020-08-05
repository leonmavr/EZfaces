try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='EZfaces',
    version='0.1',
    description='Face recognition using Eigenfaces',
    author='0xLeo',
    author_email='0xleo.git@gmail.com',
    url='https://github.com/0xLeo/EZfaces/archive/alpha.tar.gz',
    packages=['src'],
    include_package_data=True,
    install_requires=['opencv-python',
        'scikit-learn',
        'matplotlib',
        'pickle',
        'numpy'],
    zip_safe=False,
)
