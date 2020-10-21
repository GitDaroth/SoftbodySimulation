#include "Scenario/Scenario.h"

void Scenario::initialize()
{
	m_pbdSolver->clearParticles();
	m_pbdSolver->clearConstraints();

	onInitialize();

	m_pbdSolver->prepareGPUBuffers();
}

void Scenario::update(float deltaTime)
{
	onUpdate(deltaTime);

	m_pbdSolver->updateStaticCollisionGPUBuffers();
}

void Scenario::setPBDSolver(std::shared_ptr<PBDSolver> pbdSolver)
{
	m_pbdSolver = pbdSolver;
}
