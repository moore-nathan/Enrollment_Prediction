from setuptools import setup

setup(
    name='Enrollment_Prediction',
    version='1.1',
    packages=['Enrollment_Prediction'],
    url='https://github.com/whitecheeks33/Enrollment_Prediction',
    license='SFU',
    author='Nathan Moore',
    author_email='cheeksofwhite@gmail.com',
    description='Enrollment Prediction for Saint Francis University',
    install_requires = ['pandas', 'numpy',"sklearn","xgboost",'matplotlib',
              'pycountry_convert','pycountry']
)
