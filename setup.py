import setuptools


def readme():
    with open('README.rst') as f:
        return f.read()


setuptools.setup(
    name='pygma',
    version='0.2.dev1',
    author='Alexander Belinsky',
    author_email='a.v.belinsky@gmail.com',
    description='Python mathematics for deep reinforcement learning on graphs',
    long_description=readme(),
    long_description_content_type="text/x-rst",
    url='http://github.com/abelinsky',
    license='MIT',
    packages=setuptools.find_packages(),
    classifiers=[
        # https://pypi.org/classifiers/
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'numpy',
    ],
    test_suite='nose.collector',
    tests_require=['nose', 'nose-cover3'],
    python_requires='>=3.5',
    include_package_data=True,
    zip_safe=False
)
