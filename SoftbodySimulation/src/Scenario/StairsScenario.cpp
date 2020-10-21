#include "Scenario/StairsScenario.h"

#include <Constraint/BoxConstraint.h>
#include <Constraint/RigidShapeConstraint.h>
#include <Constraint/SoftShapeLinearConstraint.h>
#include <Constraint/SoftShapeQuadraticConstraint.h>

const QString StairsScenario::NAME = "Stairs";

StairsScenario::StairsScenario()
{
}

StairsScenario::~StairsScenario()
{
}

void StairsScenario::onInitialize()
{
	m_elapsedTime = 0.f;

	std::shared_ptr<ShapeConstraint> shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/bunny_small.pts", "assets/meshes/bunny_small.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(-9.f, 15.f, -3.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
	{
		particle->setVelocity(QVector3D(1.5f, 0.f, 0.f));
		m_pbdSolver->addParticle(particle);
	}

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/bunny_normal.pts", "assets/meshes/bunny_normal.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(-9.f, 15.f, 0.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
	{
		particle->setVelocity(QVector3D(1.5f, 0.f, 0.f));
		m_pbdSolver->addParticle(particle);
	}

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/bunny_big.pts", "assets/meshes/bunny_big.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(-9.f, 15.f, 3.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
	{
		particle->setVelocity(QVector3D(1.5f, 0.f, 0.f));
		m_pbdSolver->addParticle(particle);
	}

	std::shared_ptr<BoxConstraint> boxBoundaryConstraint = std::make_shared<BoxConstraint>(QVector3D(0.f, 10.f, 0.f), QVector3D(10.f, 10.f, 5.f));
	boxBoundaryConstraint->setRotation(0.f, QVector3D(1.f, 0.f, 0.f));
	boxBoundaryConstraint->setIsBoundary(true);
	m_pbdSolver->addConstraint(boxBoundaryConstraint);

	m_stairStep1 = std::make_shared<BoxConstraint>(QVector3D(-9.f, 11.f, 0.f), QVector3D(1.f, 0.75f, 4.f));
	m_stairStep1->setRotation(7.f, QVector3D(0.f, 1.f, 0.f));
	m_stairStep1->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_stairStep1);

	m_stairStep2 = std::make_shared<BoxConstraint>(QVector3D(-7.f, 9.f, 0.f), QVector3D(1.f, 0.75f, 4.f));
	m_stairStep2->setRotation(-7.f, QVector3D(0.f, 1.f, 0.f));
	m_stairStep2->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_stairStep2);

	m_stairStep3 = std::make_shared<BoxConstraint>(QVector3D(-5.f, 7.f, 0.f), QVector3D(1.f, 0.75f, 4.f));
	m_stairStep3->setRotation(7.f, QVector3D(0.f, 1.f, 0.f));
	m_stairStep3->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_stairStep3);

	m_stairStep4 = std::make_shared<BoxConstraint>(QVector3D(-3.f, 5.f, 0.f), QVector3D(1.f, 0.75f, 4.f));
	m_stairStep4->setRotation(-7.f, QVector3D(0.f, 1.f, 0.f));
	m_stairStep4->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_stairStep4);

	m_stairStep5 = std::make_shared<BoxConstraint>(QVector3D(-1.f, 3.f, 0.f), QVector3D(1.f, 0.75f, 4.f));
	m_stairStep5->setRotation(7.f, QVector3D(0.f, 1.f, 0.f));
	m_stairStep5->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_stairStep5);
}

void StairsScenario::onUpdate(float deltaTime)
{
	m_elapsedTime += deltaTime;
	m_stairStep1->setRotation(7.f * cosf(m_elapsedTime), QVector3D(0.f, 1.f, 0.f));
	m_stairStep2->setRotation(-7.f * cosf(m_elapsedTime), QVector3D(0.f, 1.f, 0.f));
	m_stairStep3->setRotation(7.f * cosf(m_elapsedTime), QVector3D(0.f, 1.f, 0.f));
	m_stairStep4->setRotation(-7.f * cosf(m_elapsedTime), QVector3D(0.f, 1.f, 0.f));
	m_stairStep5->setRotation(7.f * cosf(m_elapsedTime), QVector3D(0.f, 1.f, 0.f));
}