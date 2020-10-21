#include "Scenario/SpinningBoxesScenario.h"

#include <Constraint/BoxConstraint.h>
#include <Constraint/RigidShapeConstraint.h>
#include <Constraint/SoftShapeLinearConstraint.h>
#include <Constraint/SoftShapeQuadraticConstraint.h>

const QString SpinningBoxesScenario::NAME = "SpinningBoxes";

SpinningBoxesScenario::SpinningBoxesScenario()
{
}

SpinningBoxesScenario::~SpinningBoxesScenario()
{
}

void SpinningBoxesScenario::onInitialize()
{
	m_elapsedTime = 0.f;

	int width = 4;
	int height = 2;
	int depth = 2;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < depth; j++)
		{
			for (int k = 0; k < height; k++)
			{
				std::shared_ptr<ShapeConstraint> shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/dragon.pts", "assets/meshes/dragon.obj");
				shapeConstraint->initialize(true);
				shapeConstraint->setStartPosition(QVector3D(-2.f + i * 1.5f, 18.f + k * 2.f, -0.5f + j * 1.f));
				m_pbdSolver->addConstraint(shapeConstraint);
				for (auto particle : shapeConstraint->getParticles())
					m_pbdSolver->addParticle(particle);
			}
		}
	}

	std::shared_ptr<BoxConstraint> boxBoundaryConstraint = std::make_shared<BoxConstraint>(QVector3D(0.f, 11.f, 0.f), QVector3D(3.5f, 11.f, 1.f));
	boxBoundaryConstraint->setRotation(0.f, QVector3D(1.f, 0.f, 0.f));
	boxBoundaryConstraint->setIsBoundary(true);
	m_pbdSolver->addConstraint(boxBoundaryConstraint);

	m_spiningBox1a = std::make_shared<BoxConstraint>(QVector3D(-1.75f, 15.f, 0.f), QVector3D(1.5f, 0.25f, 1.f));
	m_spiningBox1a->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox1a->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_spiningBox1a);

	m_spiningBox1b = std::make_shared<BoxConstraint>(QVector3D(-1.75f, 15.f, 0.f), QVector3D(1.5f, 0.25f, 1.f));
	m_spiningBox1b->setRotation(90.f, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox1b->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_spiningBox1b);


	m_spiningBox2a = std::make_shared<BoxConstraint>(QVector3D(1.75f, 13.f, 0.f), QVector3D(1.5f, 0.25f, 1.f));
	m_spiningBox2a->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox2a->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_spiningBox2a);

	m_spiningBox2b = std::make_shared<BoxConstraint>(QVector3D(1.75f, 13.f, 0.f), QVector3D(1.5f, 0.25f, 1.f));
	m_spiningBox2b->setRotation(90.f, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox2b->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_spiningBox2b);


	m_spiningBox3a = std::make_shared<BoxConstraint>(QVector3D(-1.75f, 11.f, 0.f), QVector3D(1.5f, 0.25f, 1.f));
	m_spiningBox3a->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox3a->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_spiningBox3a);

	m_spiningBox3b = std::make_shared<BoxConstraint>(QVector3D(-1.75f, 11.f, 0.f), QVector3D(1.5f, 0.25f, 1.f));
	m_spiningBox3b->setRotation(90.f, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox3b->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_spiningBox3b);


	m_spiningBox4a = std::make_shared<BoxConstraint>(QVector3D(1.75f, 9.f, 0.f), QVector3D(1.5f, 0.25f, 1.f));
	m_spiningBox4a->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox4a->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_spiningBox4a);

	m_spiningBox4b = std::make_shared<BoxConstraint>(QVector3D(1.75f, 9.f, 0.f), QVector3D(1.5f, 0.25f, 1.f));
	m_spiningBox4b->setRotation(90.f, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox4b->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_spiningBox4b);


	m_spiningBox5a = std::make_shared<BoxConstraint>(QVector3D(-1.75f, 7.f, 0.f), QVector3D(1.5f, 0.25f, 1.f));
	m_spiningBox5a->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox5a->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_spiningBox5a);

	m_spiningBox5b = std::make_shared<BoxConstraint>(QVector3D(-1.75f, 7.f, 0.f), QVector3D(1.5f, 0.25f, 1.f));
	m_spiningBox5b->setRotation(90.f, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox5b->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_spiningBox5b);


	m_spiningBox6a = std::make_shared<BoxConstraint>(QVector3D(1.75f, 5.f, 0.f), QVector3D(1.5f, 0.25f, 1.f));
	m_spiningBox6a->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox6a->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_spiningBox6a);

	m_spiningBox6b = std::make_shared<BoxConstraint>(QVector3D(1.75f, 5.f, 0.f), QVector3D(1.5f, 0.25f, 1.f));
	m_spiningBox6b->setRotation(90.f, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox6b->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_spiningBox6b);


	m_spiningBox7a = std::make_shared<BoxConstraint>(QVector3D(-1.75f, 3.f, 0.f), QVector3D(1.5f, 0.25f, 1.f));
	m_spiningBox7a->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox7a->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_spiningBox7a);

	m_spiningBox7b = std::make_shared<BoxConstraint>(QVector3D(-1.75f, 3.f, 0.f), QVector3D(1.5f, 0.25f, 1.f));
	m_spiningBox7b->setRotation(90.f, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox7b->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_spiningBox7b);
}

void SpinningBoxesScenario::onUpdate(float deltaTime)
{
	m_elapsedTime += deltaTime;
	m_spiningBox1a->setRotation(-100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox1b->setRotation(90.f - 100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox2a->setRotation(100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox2b->setRotation(90.f + 100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox3a->setRotation(100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox3b->setRotation(90.f + 100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox4a->setRotation(-100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox4b->setRotation(90.f - 100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox5a->setRotation(-100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox5b->setRotation(90.f - 100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox6a->setRotation(100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox6b->setRotation(90.f + 100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox7a->setRotation(100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_spiningBox7b->setRotation(90.f + 100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
}