using System;
using System.Collections.Generic;

public class Idol
{
    private string iName;
    private int iScore;
    private int inumber;

    private Dictionary<String, int> idolUsers = new Dictionary<string, int>();

    //아이돌 명
    public string IName
    {
        get { return iName; }
        set { iName = value; }
    }

    // 누적된 총 점수를 담을 변수
    public int IScore
    {
        get { return iScore; }
        set { iScore = value; }
    }

    // 아이돌이 저장하고 있는 각 유저별 누적 응원 점수
    // 파일에 저장 할 때 유저명:점수 유저명: 점수로 저장
    public Dictionary<String, int> IdolUsers
    {
        get { return idolUsers; }
        set { idolUsers = value; }
    }

    public int Inumber
    {
        get { return inumber; }
        set { inumber = value; }
    }


    public Idol(String name)
    {
        this.iName = name;
    }



}