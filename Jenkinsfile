pipeline {
    agent any

    stages {
        stage('Install Dependencies') {
            steps {
                sh 'pip3 install --user --upgrade pip'
                sh 'pip3 install --user -r backend/requirements.txt'
            }
        }
        stage('Train and Upload Model') {
            steps {
                dir('backend') {
                    sh './train_and_upload.sh'
                }
            }
        }
    }
}
