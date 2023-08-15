pipeline {
	agent any
	   stages {
	        stage('Clone Repository') {
	        steps {
	        checkout scm
	        }
	   }
	   stage('Build Image') {
	        steps {
	        bat 'docker build -t sfm_refrigerator:v1 .'
	        }
	   }
	   stage('Run Image') {
	        steps {
	        bat 'docker run -p 5000:8000 -d --name sfm_refrigerator sfm_refrigerator:v1'
	        }
	   }
	   stage('Testing'){
	        steps {
	            echo 'Testing..'
	            }
	   }
    }
}