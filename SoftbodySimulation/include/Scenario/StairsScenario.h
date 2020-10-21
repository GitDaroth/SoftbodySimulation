#pragma once

#include "Scenario/Scenario.h"

class StairsScenario : public Scenario
{
	Q_OBJECT

public:
	StairsScenario();
	~StairsScenario();

	virtual void onInitialize() override;
	virtual void onUpdate(float deltaTime) override;
	static const QString NAME;

private:
	float m_elapsedTime;
	std::shared_ptr<BoxConstraint> m_stairStep1;
	std::shared_ptr<BoxConstraint> m_stairStep2;
	std::shared_ptr<BoxConstraint> m_stairStep3;
	std::shared_ptr<BoxConstraint> m_stairStep4;
	std::shared_ptr<BoxConstraint> m_stairStep5;
};