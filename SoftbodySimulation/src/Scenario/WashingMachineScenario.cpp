#include "Scenario/WashingMachineScenario.h"

#include <Constraint/BoxConstraint.h>
#include <Constraint/RigidShapeConstraint.h>
#include <Constraint/SoftShapeLinearConstraint.h>
#include <Constraint/SoftShapeQuadraticConstraint.h>

const QString WashingMachineScenario::NAME = "Washing Machine";

WashingMachineScenario::WashingMachineScenario()
{
}

WashingMachineScenario::~WashingMachineScenario()
{
}

void WashingMachineScenario::onInitialize()
{
	m_elapsedTime = 0.f;

	int height = 2;
	int width = 5;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (j % 2 == 0)
			{
				std::shared_ptr<ShapeConstraint> shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/armadillo.pts", "assets/meshes/armadillo.obj");
				shapeConstraint->initialize(true);
				shapeConstraint->setStartPosition(QVector3D(-2.5f, 7.07f + 2.f * i, 2.f * (j - 2)));
				m_pbdSolver->addConstraint(shapeConstraint);
				for (auto particle : shapeConstraint->getParticles())
					m_pbdSolver->addParticle(particle);

				if (i > 0)
				{
					shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/bunny_big.pts", "assets/meshes/bunny_big.obj");
					shapeConstraint->initialize(true);
					shapeConstraint->setStartPosition(QVector3D(0.f, 7.07f + 2.f * i, 2.f * (j - 2)));
					m_pbdSolver->addConstraint(shapeConstraint);
					for (auto particle : shapeConstraint->getParticles())
						m_pbdSolver->addParticle(particle);
				}

				shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/octopus2.pts", "assets/meshes/octopus2.obj");
				shapeConstraint->initialize(true);
				shapeConstraint->setStartPosition(QVector3D(2.5f, 7.07f + 2.f * i, 2.f * (j - 2)));
				m_pbdSolver->addConstraint(shapeConstraint);
				for (auto particle : shapeConstraint->getParticles())
					m_pbdSolver->addParticle(particle);
			}	
			else
			{
				std::shared_ptr<ShapeConstraint> shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/dragon.pts", "assets/meshes/dragon.obj");
				shapeConstraint->initialize(true);
				shapeConstraint->setStartPosition(QVector3D(-2.5f, 7.07f + 2.f * i, 2.f * (j - 2)));
				m_pbdSolver->addConstraint(shapeConstraint);
				for (auto particle : shapeConstraint->getParticles())
					m_pbdSolver->addParticle(particle);

				if (i > 0)
				{
					shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/bunny_normal.pts", "assets/meshes/bunny_normal.obj");
					shapeConstraint->initialize(true);
					shapeConstraint->setStartPosition(QVector3D(0.f, 7.07f + 2.f * i, 2.f * (j - 2)));
					m_pbdSolver->addConstraint(shapeConstraint);
					for (auto particle : shapeConstraint->getParticles())
						m_pbdSolver->addParticle(particle);
				}

				shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/patrick.pts", "assets/meshes/patrick.obj");
				shapeConstraint->initialize(true);
				shapeConstraint->setStartPosition(QVector3D(2.5f, 7.07f + 2.f * i, 2.f * (j - 2)));
				m_pbdSolver->addConstraint(shapeConstraint);
				for (auto particle : shapeConstraint->getParticles())
					m_pbdSolver->addParticle(particle);
			}
		}
	}

	m_washingMachine = std::make_shared<BoxConstraint>(QVector3D(0.f, 7.07f, 0.f), QVector3D(5.f, 5.f, 5.f));
	m_washingMachine->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	m_washingMachine->setIsBoundary(true);
	m_pbdSolver->addConstraint(m_washingMachine);

	m_innerBox = std::make_shared<BoxConstraint>(QVector3D(0.f, 7.07f, 0.f), QVector3D(1.f, 1.f, 5.f));
	m_innerBox->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	m_innerBox->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_innerBox);
}

void WashingMachineScenario::onUpdate(float deltaTime)
{
	m_elapsedTime += deltaTime;
	m_washingMachine->setRotation(100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_innerBox->setRotation(-100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
}