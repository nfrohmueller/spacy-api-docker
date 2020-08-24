#!groovy

library identifier: 'sparks-jenkins-library@master',
        retriever: modernSCM([$class: 'GitSCMSource', remote: 'git@github.com:rewe-digital-content/sparks-jenkins-library.git', credentialsId: 'global-scm-key-credentials']),
        changelog: false

def config = [intValuesFile:"./service-values-int.yaml",
              prdValuesFile:"./service-values-prd.yaml"
]

rdServicePipeline(config)
