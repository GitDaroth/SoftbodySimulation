#pragma once

#include "Scenario/Scenario.h"

class SpinningBoxesScenario : public Scenario
{
	Q_OBJECT

public:
	SpinningBoxesScenario();
	~SpinningBoxesScenario();

	virtual void onInitialize() override;
	virtual void onUpdate(float deltaTime) override;
	static const QString NAME;

private:
	float m_elapsedTime;
	std::shared_ptr<BoxConstraint> m_spiningBox1a;
	std::shared_ptr<BoxConstraint> m_spiningBox1b;
	std::shared_ptr<BoxConstraint> m_spiningBox2a;
	std::shared_ptr<BoxConstraint> m_spiningBox2b;
	std::shared_ptr<BoxConstraint> m_spiningBox3a;
	std::shared_ptr<BoxConstraint> m_spiningBox3b;
	std::shared_ptr<BoxConstraint> m_spiningBox4a;
	std::shared_ptr<BoxConstraint> m_spiningBox4b;
	std::shared_ptr<BoxConstraint> m_spiningBox5a;
	std::shared_ptr<BoxConstraint> m_spiningBox5b;
	std::shared_ptr<BoxConstraint> m_spiningBox6a;
	std::shared_ptr<BoxConstraint> m_spiningBox6b;
	std::shared_ptr<BoxConstraint> m_spiningBox7a;
	std::shared_ptr<BoxConstraint> m_spiningBox7b;
};