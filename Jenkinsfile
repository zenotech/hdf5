node('c++') {
  stage 'Build and Test'
  checkout scm
  dir('build') {
    sh '''cmake ...
          make
          make test'''
  }
}