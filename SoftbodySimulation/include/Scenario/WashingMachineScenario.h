#pragma once

#include "Scenario/Scenario.h"

class WashingMachineScenario : public Scenario
{
	Q_OBJECT

public:
	WashingMachineScenario();
	~WashingMachineScenario();

	virtual void onInitialize() override;
	virtual void onUpdate(float deltaTime) override;
	static const QString NAME;

private:
	float m_elapsedTime;
	std::shared_ptr<BoxConstraint> m_washingMachine;
	std::shared_ptr<BoxConstraint> m_innerBox;
};