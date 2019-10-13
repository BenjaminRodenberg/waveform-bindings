from setuptools import setup, find_packages

setup(name='waveform-bindings',
      version='0.1',
      description='wrapper for the preCICE python bindings to add waveform relaxation',
      url='https://github.com/BenjaminRueth/waveform-bindings',
      author="Benjamin Rueth",
      author_email='benjamin.rueth@tum.de',
      license='LGPL-3.0',
      packages=['waveformbindings'],
      install_requires=['precice_future', 'scipy', 'numpy>=1.13.3'],
      test_suite='tests',
      zip_safe=False)
