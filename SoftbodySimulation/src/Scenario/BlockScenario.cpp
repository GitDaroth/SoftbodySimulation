#include "Scenario/BlockScenario.h"

#include <Constraint/BoxConstraint.h>
#include <Constraint/RigidShapeConstraint.h>
#include <Constraint/SoftShapeLinearConstraint.h>
#include <Constraint/SoftShapeQuadraticConstraint.h>

const QString BlockScenario::NAME = "Block";

BlockScenario::BlockScenario()
{
}

BlockScenario::~BlockScenario()
{
}

void BlockScenario::onInitialize()
{
	m_elapsedTime = 0.f;

	std::shared_ptr<ShapeConstraint> shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/bunny_normal.pts", "assets/meshes/bunny_normal.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(-8.f, 18.f, 0.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/octopus2.pts", "assets/meshes/octopus2.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(-5.5f, 18.f, 0.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/octopus2.pts", "assets/meshes/octopus2.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(-2.6f, 18.f, 0.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/dragon.pts", "assets/meshes/dragon.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(0.f, 18.f, 0.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/armadillo.pts", "assets/meshes/armadillo.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(2.6f, 18.f, 0.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/armadillo.pts", "assets/meshes/armadillo.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(5.5f, 18.f, 0.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/patrick.pts", "assets/meshes/patrick.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(8.f, 18.f, 0.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);


	std::shared_ptr<BoxConstraint> boxBoundaryConstraint = std::make_shared<BoxConstraint>(QVector3D(0.f, 11.f, 0.f), QVector3D(10.f, 11.f, 1.5f));
	boxBoundaryConstraint->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	boxBoundaryConstraint->setIsBoundary(true);
	m_pbdSolver->addConstraint(boxBoundaryConstraint);

	m_blocks.clear();

	for (int i = 0; i < 3; i++)
	{
		if (i % 2 == 0)
		{
			for (int j = 0; j < 4; j++)
			{
				std::shared_ptr<BoxConstraint> block = std::make_shared<BoxConstraint>(QVector3D(-7.875f + 5.25f * j, 5.f + 5.f * i, 0.f), QVector3D(1.5f, 1.5f, 1.5f));
				block->setRotation(45.f, QVector3D(0.f, 0.f, 1.f));
				block->setIsBoundary(false);
				m_pbdSolver->addConstraint(block);

				m_blocks.push_back(block);
			}
		}
		else
		{
			for (int j = 0; j < 3; j++)
			{
				std::shared_ptr<BoxConstraint> block = std::make_shared<BoxConstraint>(QVector3D(-5.25f + 5.25f * j, 5.f + 5.f * i, 0.f), QVector3D(1.5f, 1.5f, 1.5f));
				block->setRotation(45.f, QVector3D(0.f, 0.f, 1.f));
				block->setIsBoundary(false);
				m_pbdSolver->addConstraint(block);

				m_blocks.push_back(block);
			}
		}
	}
}

void BlockScenario::onUpdate(float deltaTime)
{
	m_elapsedTime += deltaTime;
	m_blocks[2]->setRotation(500.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_blocks[4]->setRotation(-500.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_blocks[10]->setRotation(500.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
}