# Reinforcement Learning for Smart Route Selection

This project is about teaching a computer to make smart choices.
Imagine you have three helpers that can answer a question:

- Hybrid RAG
- GraphRAG
- Optimized GraphRAG

The job of the learning system is to choose the best helper for each question.
It learns by trying, seeing what worked, and getting better over time.
That is reinforcement learning.

## The big idea

Think of it like teaching a child to choose the best path to school.

- Some roads are fast but bumpy.
- Some roads are slow but safe.
- Some roads are just right.

The learner tries each route, sees how good the result was, and learns which
route is usually the best trade-off between quality and cost.

## What the RL part does

The project trains a router.
A router is a small decision-maker.
It looks at a question and decides which retrieval route to use first.

It does not try to answer the question itself.
It only picks the best tool to use.

## What algorithm is used?

The main algorithm is a contextual bandit.

A bandit is a simple kind of reinforcement learning.
It learns by trying actions and getting rewards.

In this project, each action is one route:

- choose Hybrid RAG
- choose GraphRAG
- choose Optimized GraphRAG

The router receives a small set of question features, such as:

- question length
- number of named entities
- words that suggest relations or multi-step thinking
- hints that the question may need graph reasoning

These features are like clues that help the router make a guess.

## How the training works

Training happens in three simple steps.

### 1. Collect examples

For each training question, the project runs all three routes.
That means it sees what happens if it chose each option.

This is important because the system learns from full feedback.
It can compare:

- which route was cheap
- which route was strong
- which route gave the best balance

### 2. Give each route a reward

Each route gets a score based on two things:

- quality: did it help find good evidence or answer content?
- cost: how much time, space, and work did it use?

The system wants a route that is both good and efficient.
So it rewards routes that are strong without wasting too much energy.

### 3. Learn a policy

The router learns a policy.
A policy is just a rule for choosing an action.

The policy learns to say things like:

- “For short questions, Hybrid RAG is often enough.”
- “For questions with many relations, GraphRAG may be better.”
- “For tricky multi-hop questions, Optimized GraphRAG may be the sweet spot.”

## Why this is useful

Some questions are easy.
Some are hard.
Some need more thinking.

Using the strongest route every time can be too expensive.
Using the cheapest route every time can be too weak.

The RL router learns to strike a balance.
It tries to be smart, not just fast or strong.

## A simple analogy

This project is like having a robot helper at a school fair.

- One helper is fast but only knows simple questions.
- One helper is strong but uses a lot of energy.
- One helper is careful and works well for tricky questions.

The robot learns which helper to call for each job.
That is the heart of the reinforcement learning idea here.

## What makes this project special

The project uses offline reinforcement learning-style training.
That means it learns from a prepared set of examples instead of learning only
from live user feedback.

This is safer and easier to study.
It gives the router a strong starting point before it is used more broadly.

## What comes next

The current router chooses one route first.
In the future, it could learn more advanced behavior, such as:

- try one route
- check how good the result looks
- decide whether to stop, escalate, or switch

That would be a more advanced step toward intelligent decision-making.

## In one sentence

This project teaches a smart router to pick the best question-answering path by
learning from rewards, just like a student learns which study method works best.
