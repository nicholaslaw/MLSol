from setuptools import setup

setup(
   name='MLSol',
   version='1.0.0',
   description='A module to perform multilabel oversampling based on "Synthetic Oversampling of Multi-Label Data based on Local Label Distribution" by Bin Liu and d Grigorios Tsoumakas',
   author='Nicholas Law',
   author_email='nicholas_law_91@hotmail.com',
   packages=['MLSol'],  #same as name
   install_requires=[
       "numpy==1.18.1"
   ], #external packages as dependencies
)