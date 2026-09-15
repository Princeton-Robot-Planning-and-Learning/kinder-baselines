(define (domain limbrepositioning3d)
  (:requirements :strips :equality)
  (:predicates
    ; Static types.
    (Limb ?l)
    (Grasp ?l ?g)                ; ?g maps the limb grasp frame to the end effector
    (BConf ?q)                   ; an SE2 pose of the mobile base
    (Conf ?qL)                   ; a passive-limb joint configuration
    (InitConf ?l ?qL)
    (GoalConf ?l ?qL)
    (State ?s)                   ; base pose plus robot and limb joints and velocities
    (StateConf ?s ?qL)           ; the limb configuration within state ?s
    (Reachable ?l ?g ?q ?qL)     ; from base ?q the arm can hold ?g at limb conf ?qL
    (BaseMotion ?q1 ?bt ?q2)     ; base path ?bt drives the base from ?q1 to ?q2
    (GraspKin ?l ?g ?q ?at ?s)   ; arm path ?at from base ?q reaches ?g, leaving ?s
    (LimbMotion ?s1 ?qL2 ?tt ?s2) ; torques ?tt drive ?s1 to ?s2, with limb conf ?qL2
    (SafeRobotTorques ?tt)       ; every torque in ?tt is within the robot's limits
    (SafeHumanTorques ?tt)       ; ?tt never overloads a joint of the person
    (SafeHumanJoints ?tt)        ; ?tt keeps the limb in the person's range of motion

    ; Fluents.
    (AtBConf ?q)
    (AtState ?s)
    (Grasped ?l ?g)              ; ?l is held with grasp ?g, pinning later reachability
    (HandEmpty)
    (CanMove)                    ; guards against consecutive move_base actions

    ; Derived.
    (AtConf ?qL)
  )

  ; Only before grasping: driving while welded would drag the limb through the scene.
  (:action move_base
    :parameters (?q1 ?bt ?q2)
    :precondition (and (BaseMotion ?q1 ?bt ?q2)
                       (AtBConf ?q1) (CanMove) (HandEmpty))
    :effect (and (AtBConf ?q2)
                 (not (AtBConf ?q1)) (not (CanMove)))
  )

  ; CanMove is not restored, so the base stays put once the limb is grasped.
  (:action grasp
    :parameters (?l ?g ?q ?at ?s)
    :precondition (and (GraspKin ?l ?g ?q ?at ?s)
                       (AtBConf ?q) (HandEmpty))
    :effect (and (Grasped ?l ?g) (AtState ?s)
                 (not (HandEmpty)))
  )

  ; `Reachable` ties this action to where the base was parked, so the planner cannot
  ; commit to a pose it can grasp from but never finish from.
  (:action move_limb
    :parameters (?l ?g ?q ?s1 ?qL1 ?qL2 ?tt ?s2)
    :precondition (and (Grasped ?l ?g)
                       (AtState ?s1)
                       (StateConf ?s1 ?qL1)
                       (not (= ?qL1 ?qL2))
                       (AtBConf ?q)
                       (Reachable ?l ?g ?q ?qL2)
                       (LimbMotion ?s1 ?qL2 ?tt ?s2)
                       (SafeRobotTorques ?tt)
                       (SafeHumanTorques ?tt)
                       (SafeHumanJoints ?tt))
    :effect (and (AtState ?s2)
                 (not (AtState ?s1)))
  )

  (:derived (AtConf ?qL)
    (exists (?s) (and (State ?s)
                      (Conf ?qL)
                      (StateConf ?s ?qL)
                      (AtState ?s)))
  )
)
