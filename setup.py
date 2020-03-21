from setuptools import setup, find_packages

setup(
    name='Enrollment_Prediction',
    version='1.1',
    packages=find_packages(exclude='Data'),
    url='https://github.com/whitecheeks33/Enrollment_Prediction',
    license='SFU',
    author='Nathan Moore',
    author_email='cheeksofwhite@gmail.com',
    description='Enrollment Prediction for Saint Francis University',
    install_requires=['pandas', 'numpy', "sklearn", "xgboost", 'matplotlib',
                      'pycountry_convert', 'pycountry'],
    entry_points={'console_scripts': [
            'runner= Enrollment_Prediction.enrollment_prediction:main',
        ],
    },
)
