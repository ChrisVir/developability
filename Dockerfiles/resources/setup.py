from setuptools import setup, find_packages


entry = 'dev_models.entrypoints'

setup(name='developability_models',
      description='''Simple module for running models for developability
                     in docker''',
      author='Christopher Rivera',
      packages=find_packages(include=['dev_models', 'dev_models.*']),
      entry_points={
          'console_scripts': [
              f'predict_heparin_binding = {entry}:predict_heparin_binding'
              ]
      }

      )
