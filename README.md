# ASLAI — Reconhecimento e Tradução de Língua Gestual

## Sobre o Projeto

Este projeto tem como objetivo o desenvolvimento de um sistema capaz de reconhecer gestos de língua gestual e convertê-los em texto e voz, recorrendo a técnicas de **Visão Computacional** e **Machine Learning**.

A motivação principal do trabalho é contribuir para a **redução das barreiras de comunicação** entre a comunidade surda e a população ouvinte, explorando soluções tecnológicas acessíveis, não invasivas e com potencial de aplicação em contextos reais.

## Tema

O projeto enquadra-se na área de **Interação Pessoa-Máquina**, combinando:

- captura de vídeo em tempo real;
- extração de landmarks com **MediaPipe**;
- reconhecimento de gestos estáticos e dinâmicos;
- conversão dos gestos reconhecidos em **texto**;
- síntese da frase final em **voz**.

Embora a motivação original esteja centrada na **Língua Gestual Portuguesa (LGP)**, a implementação prática recorre à **American Sign Language (ASL)**, devido à maior disponibilidade de datasets públicos para treino e validação dos modelos.

## Objetivo Principal

Criar uma solução inteligente e acessível que permita:

- reconhecer gestos feitos pelo utilizador através da câmara;
- interpretar esses gestos com recurso a modelos de Machine Learning;
- apresentar o resultado em formato textual;
- produzir também uma saída em áudio, tornando a comunicação mais natural.

## Abordagem do Sistema

O sistema foi pensado em torno de dois tipos de reconhecimento:

### 1. Gestos Estáticos
Reconhecimento de letras do alfabeto gestual a partir de uma única frame, usando landmarks da mão e modelos de classificação.

### 2. Gestos Dinâmicos
Reconhecimento de palavras ou sequências gestuais com componente temporal, usando sequências de frames e modelos adequados a dados temporais.

## Tecnologias e Conceitos Envolvidos

- **MediaPipe**
- **Visão Computacional**
- **Machine Learning**
- **MLP**
- **LSTM**
- **Landmarks 3D**
- **Reconhecimento de gestos**
- **Texto para voz**
- **Acessibilidade**
- **Inteligência Artificial aplicada à inclusão**

## Motivação

A língua gestual é um meio de comunicação essencial para muitas pessoas surdas. No entanto, continuam a existir dificuldades de comunicação em vários contextos do quotidiano, como:

- educação;
- saúde;
- serviços públicos;
- transportes;
- atendimento geral.

Este projeto surge como uma proposta tecnológica para aproximar mundos que muitas vezes continuam separados por falta de ferramentas acessíveis e imediatas.

## Impacto Esperado

Espera-se que este tipo de sistema possa contribuir para:

- melhorar a acessibilidade;
- promover inclusão social;
- facilitar a comunicação em tempo real;
- apoiar contextos educativos e institucionais;
- demonstrar o potencial da IA em problemas com impacto humano real.

---

**Universidade do Minho**  
**Mestrado em Inteligência Artificial**  
**Sensorização e Ambiente**










