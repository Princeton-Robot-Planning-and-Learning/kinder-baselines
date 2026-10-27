(define (domain masstransport3d)
  (:requirements :strips :equality :action-costs)
  (:constants src dst)
  (:predicates
    ; Static facts, from the initial state or certified by streams.
    (Item ?o)
    (Slot ?s)
    (Pose ?o ?p)
    (PoseIn ?p ?r)
    (DestPose ?o ?p)
    (Grasp ?o ?g)
    (TotePose ?tp)
    (TotePoseIn ?tp ?r)
    (ToteGrasp ?g)
    (CarryConf ?g ?a)
    (BConf ?q)
    (BConfIn ?q ?r)
    (TableBase ?q ?r)
    (ATraj ?t)
    (Kin ?o ?p ?g ?q ?t)
    (SlotKin ?o ?s ?tp ?g ?q ?t)
    (ToteKin ?tp ?g ?a ?q ?t)
    (BaseMotion ?q1 ?t ?q2)
    (NextTrip ?n1 ?n2)
    (CFreePosePose ?o ?p ?o2 ?p2)
    (CFreePoseTote ?o ?p ?tp)
    (CFreeTrajPose ?t ?o2 ?p2)
    (CFreeTrajTote ?t ?tp)
    ; Certified by a fluent test: the arm can hold the tote and its contents.
    (CarryOK ?g ?a)

    ; Fluents.
    (AtPose ?o ?p)
    (AtGrasp ?o ?g)
    (InSlot ?o ?s)
    (SlotFree ?s)
    (AtTotePose ?tp)
    (HoldingTote ?g ?a)
    (AtBConf ?q)
    (HandEmpty)
    (CanMove)
    (TripsLeft ?n)

    ; Derived.
    (OnDest ?o)
    (UnsafePose ?o ?p)
    (UnsafeTotePose ?tp)
    (UnsafeTraj ?t)
  )
  (:functions
    (TravelCost)
    (total-cost)
  )

  ; Base moves stay within one region; crossing from src to dst is a trip.
  (:action move_base
    :parameters (?q1 ?q2 ?r ?t)
    :precondition (and (BaseMotion ?q1 ?t ?q2) (BConfIn ?q1 ?r) (BConfIn ?q2 ?r)
                       (AtBConf ?q1) (CanMove))
    :effect (and (AtBConf ?q2) (not (AtBConf ?q1)) (not (CanMove))
                 (increase (total-cost) 1))
  )

  (:action travel
    :parameters (?q1 ?q2 ?t ?n1 ?n2)
    :precondition (and (BaseMotion ?q1 ?t ?q2) (BConfIn ?q1 src) (BConfIn ?q2 dst)
                       (AtBConf ?q1) (TripsLeft ?n1) (NextTrip ?n1 ?n2))
    :effect (and (AtBConf ?q2) (TripsLeft ?n2) (CanMove)
                 (not (AtBConf ?q1)) (not (TripsLeft ?n1))
                 (increase (total-cost) (TravelCost)))
  )

  (:action return
    :parameters (?q1 ?q2 ?t)
    :precondition (and (BaseMotion ?q1 ?t ?q2) (BConfIn ?q1 dst) (BConfIn ?q2 src)
                       (AtBConf ?q1))
    :effect (and (AtBConf ?q2) (CanMove) (not (AtBConf ?q1))
                 (increase (total-cost) (TravelCost)))
  )

  ; Items on tables.
  (:action pick
    :parameters (?o ?p ?g ?q ?t)
    :precondition (and (Kin ?o ?p ?g ?q ?t)
                       (AtPose ?o ?p) (HandEmpty) (AtBConf ?q)
                       (not (UnsafeTraj ?t)))
    :effect (and (AtGrasp ?o ?g) (CanMove)
                 (not (AtPose ?o ?p)) (not (HandEmpty))
                 (increase (total-cost) 1))
  )

  (:action place
    :parameters (?o ?p ?g ?q ?t)
    :precondition (and (Kin ?o ?p ?g ?q ?t) (DestPose ?o ?p)
                       (AtGrasp ?o ?g) (AtBConf ?q)
                       (not (UnsafePose ?o ?p))
                       (not (UnsafeTraj ?t)))
    :effect (and (AtPose ?o ?p) (HandEmpty) (CanMove)
                 (not (AtGrasp ?o ?g))
                 (increase (total-cost) 1))
  )

  ; Items in and out of the tote's slots.
  (:action load
    :parameters (?o ?s ?tp ?g ?q ?t)
    :precondition (and (SlotKin ?o ?s ?tp ?g ?q ?t)
                       (AtGrasp ?o ?g) (AtTotePose ?tp) (SlotFree ?s) (AtBConf ?q)
                       (not (UnsafeTraj ?t)))
    :effect (and (InSlot ?o ?s) (HandEmpty) (CanMove)
                 (not (AtGrasp ?o ?g)) (not (SlotFree ?s))
                 (increase (total-cost) 1))
  )

  (:action unload
    :parameters (?o ?s ?tp ?g ?q ?t)
    :precondition (and (SlotKin ?o ?s ?tp ?g ?q ?t)
                       (InSlot ?o ?s) (AtTotePose ?tp) (HandEmpty) (AtBConf ?q)
                       (not (UnsafeTraj ?t)))
    :effect (and (AtGrasp ?o ?g) (SlotFree ?s) (CanMove)
                 (not (InSlot ?o ?s)) (not (HandEmpty))
                 (increase (total-cost) 1))
  )

  ; The tote. CarryOK checks the holding torques with the current contents.
  (:action pick_tote
    :parameters (?tp ?g ?a ?q ?t)
    :precondition (and (ToteKin ?tp ?g ?a ?q ?t) (CarryOK ?g ?a)
                       (AtTotePose ?tp) (HandEmpty) (AtBConf ?q)
                       (not (UnsafeTraj ?t)))
    :effect (and (HoldingTote ?g ?a) (CanMove)
                 (not (AtTotePose ?tp)) (not (HandEmpty))
                 (increase (total-cost) 1))
  )

  (:action place_tote
    :parameters (?tp ?g ?a ?q ?t)
    :precondition (and (ToteKin ?tp ?g ?a ?q ?t) (CarryOK ?g ?a)
                       (HoldingTote ?g ?a) (AtBConf ?q)
                       (not (UnsafeTotePose ?tp))
                       (not (UnsafeTraj ?t)))
    :effect (and (AtTotePose ?tp) (HandEmpty) (CanMove)
                 (not (HoldingTote ?g ?a))
                 (increase (total-cost) 1))
  )

  (:derived (OnDest ?o)
    (exists (?p) (and (DestPose ?o ?p) (AtPose ?o ?p)))
  )

  (:derived (UnsafePose ?o ?p)
    (or
      (exists (?o2 ?p2) (and (Pose ?o ?p) (Pose ?o2 ?p2) (not (= ?o ?o2))
                             (not (CFreePosePose ?o ?p ?o2 ?p2))
                             (AtPose ?o2 ?p2)))
      (exists (?tp) (and (Pose ?o ?p) (TotePose ?tp)
                         (not (CFreePoseTote ?o ?p ?tp))
                         (AtTotePose ?tp))))
  )
  (:derived (UnsafeTotePose ?tp)
    (exists (?o ?p) (and (TotePose ?tp) (Pose ?o ?p)
                         (not (CFreePoseTote ?o ?p ?tp))
                         (AtPose ?o ?p)))
  )
  (:derived (UnsafeTraj ?t)
    (or
      (exists (?o2 ?p2) (and (ATraj ?t) (Pose ?o2 ?p2)
                             (not (CFreeTrajPose ?t ?o2 ?p2))
                             (AtPose ?o2 ?p2)))
      (exists (?tp) (and (ATraj ?t) (TotePose ?tp)
                         (not (CFreeTrajTote ?t ?tp))
                         (AtTotePose ?tp))))
  )
)
