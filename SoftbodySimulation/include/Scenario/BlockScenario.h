#pragma once

#include "Scenario/Scenario.h"

class BlockScenario : public Scenario
{
	Q_OBJECT

public:
	BlockScenario();
	~BlockScenario();

	virtual void onInitialize() override;
	virtual void onUpdate(float deltaTime) override;
	static const QString NAME;

private:
	float m_elapsedTime;
	QVector<std::shared_ptr<BoxConstraint>> m_blocks;
};
