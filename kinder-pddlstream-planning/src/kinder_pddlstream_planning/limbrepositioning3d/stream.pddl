(define (stream limbrepositioning3d)
  ; Single-shot for now: the scene ships one fixed transform per limb family.
  (:stream sample-grasp
    :inputs (?l)
    :domain (Limb ?l)
    :outputs (?g)
    :certified (Grasp ?l ?g)
  )
  (:stream plan-grasp-motion
    :inputs (?l ?g ?qL1 ?qL2)
    :domain (and (Grasp ?l ?g) (InitConf ?l ?qL1) (GoalConf ?l ?qL2))
    :outputs (?q ?at ?s)
    :certified (and (BConf ?q)
                    (Reachable ?l ?g ?q ?qL1)
                    (Reachable ?l ?g ?q ?qL2)
                    (State ?s)
                    (StateConf ?s ?qL1)
                    (GraspKin ?l ?g ?q ?at ?s))
  )

  (:stream plan-base-motion
    :inputs (?q1 ?q2)
    :domain (and (BConf ?q1) (BConf ?q2))
    :outputs (?bt)
    :certified (BaseMotion ?q1 ?bt ?q2)
  )

  ; Runs predictive-sampling MPC and returns the open-loop torques it found.
  (:stream plan-limb-motion
    :inputs (?s1 ?qL2)
    :domain (and (State ?s1) (Conf ?qL2))
    :outputs (?tt ?s2)
    :certified (and (State ?s2)
                    (StateConf ?s2 ?qL2)
                    (LimbMotion ?s1 ?qL2 ?tt ?s2))
  )

  ; Checks the person's range of motion at every configuration ?tt passes through.
  (:stream check-human-joint-limits
    :inputs (?s1 ?qL2 ?tt ?s2)
    :domain (LimbMotion ?s1 ?qL2 ?tt ?s2)
    :outputs ()
    :certified (SafeHumanJoints ?tt)
  )

  ; Checks the torque ?tt puts on the person's own joints, measured by inverse dynamics.
  (:stream check-human-torque-limits
    :inputs (?s1 ?qL2 ?tt ?s2)
    :domain (LimbMotion ?s1 ?qL2 ?tt ?s2)
    :outputs ()
    :certified (SafeHumanTorques ?tt)
  )

  ; Checks whether the limb manipulation trajectory satisfies the robot's torque limits.
  (:stream check-robot-torque-limits
    :inputs (?s1 ?qL2 ?tt ?s2)
    :domain (LimbMotion ?s1 ?qL2 ?tt ?s2)
    :outputs ()
    :certified (SafeRobotTorques ?tt)
  )
)
