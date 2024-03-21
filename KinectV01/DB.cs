using MySql.Data.MySqlClient;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Effects;
using System.Windows.Forms;
using KinectV01;
using System.Data;

namespace FirstPage
{
    public class DB
    {
        private MySqlConnection connection;
        private string strconn = "Server=211.112.84.57;port=3307;Database=ke;Uid=root;Pwd=rootpw";


        public List<Idol> getIdolsFromDB()                                  //아이돌 리스트 생성
        {
            MySqlConnection connection = new MySqlConnection(strconn);
            string query = "SELECT idol.idol_no,idol.idol_name,user.user_no,score.score_1,user.user_name FROM idol LEFT JOIN score ON score.idol_no = idol.idol_no LEFT JOIN user ON user.user_no = score.user_no ;";
            MySqlCommand cmd = new MySqlCommand(query, connection);
            List<Idol> resultIdols = new List<Idol>();


            try
            {
                connection.Open();
                using (MySqlDataReader reader = cmd.ExecuteReader())
                {
                    while (reader.Read())
                    { 
                        //필요한 것 : idol_name, user_name, score, idol_no


                        String idolName = reader["idol_name"].ToString();
                        String userName = reader["user_name"].ToString();
                        int idolnum = int.Parse(reader["idol_no"].ToString());
                        int userScore = int.Parse(reader["score_1"].ToString());
                        bool isIdolExist = false;
                        Idol existIdol = null;

                        foreach (var idol in resultIdols)
                        {
                            if (idol.IName.Equals(idolName))
                            {
                                isIdolExist = true;
                                existIdol = idol;
                                break;
                            }
                        }

                        if (isIdolExist)
                        {
                            existIdol.Inumber = idolnum;
                            if (existIdol.IdolUsers.ContainsKey(userName))
                            {
                                existIdol.IdolUsers[userName] += userScore;
                                existIdol.IScore += userScore;
                            }
                            else
                            {
                                existIdol.IdolUsers.Add(userName, userScore);
                                existIdol.IScore += userScore;
                            }


                        }
                        else
                        {
                            Idol idol = new Idol(idolName);
                            idol.Inumber = idolnum;
                           idol.IdolUsers.Add(userName, userScore);
                            idol.IScore += userScore;
                            resultIdols.Add(idol);
                        }
                    }
                    reader.Close();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            return resultIdols;

        }


        public List<User> getUserFromDB()                              //유저 리스트 생성
        {
            MySqlConnection connection = new MySqlConnection(strconn);
            string query = "SELECT user.user_no,user.user_name,idol.idol_name,idol.idol_no,score.score_1 FROM user LEFT JOIN score ON user.user_no = score.user_no LEFT JOIN idol ON score.idol_no = idol.idol_no;";
            MySqlCommand cmd = new MySqlCommand(query, connection);
            List<User> resultUsers = new List<User>();


            try
            {
                connection.Open();
                using (MySqlDataReader reader = cmd.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        String idolName = reader["idol_name"].ToString();
                        String userName = reader["user_name"].ToString();
                        int usernum = int.Parse(reader["user_no"].ToString());
                        int userScore = int.Parse(reader["score_1"].ToString());
                        bool isUserExist = false;
                        User existUser = null;

                        foreach (var user in resultUsers)
                        {
                            if (user.UName.Equals(userName))
                            {
                                isUserExist = true;
                                existUser = user;
                                break;
                            }
                        }

                        if (isUserExist)
                        {
                            if (existUser.UserIdols.ContainsKey(idolName))
                            {
                                existUser.UserIdols[idolName] += userScore;
                                existUser.UScore += userScore;
                            }
                            else
                            {
                                existUser.UserIdols.Add(idolName, userScore);
                                existUser.UScore += userScore;
                            }

                        }
                        else
                        {
                            User User = new User();
                            User.Unumber = usernum;
                            User.UName = userName;
                            User.UserIdols.Add(idolName, userScore);
                            User.UScore += userScore;
                            resultUsers.Add(User);
                        }
                    }
                    reader.Close();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            return resultUsers;

        }

        //EnterName에서 호출 유저이름, 아이돌이름, 아이돌리스트, 유저리스트를 받고
        //DB에 상황에 맞게 데이터 저장 후
        //'유저명 아이돌명' 형태의 스트링 반환
        public Tuple<Idol,User> SetUserAndIdol(String userName, String IdolName, List<Idol> idolList, List<User> userList)
        {

            bool isUserExist = false;
            bool isIdolExist = false;
            Idol resultIdol = null;
            User resultUser = null;

            foreach (var user in userList)                     //유저명이 존재하면 true
            {
                if (user.UName.Equals(userName))
                {
                    isUserExist = true;
                    resultUser = user;
                    break;
                }
            }

            foreach (var idol in idolList)                    //아이돌명이 존재하면 true
            {
                if (idol.IName.Equals(IdolName))
                {
                    isIdolExist = true;
                    resultIdol = idol;
                    break;
                }
            }


            if (isUserExist && isIdolExist)                     //둘다 있으면
            {
                MessageBox.Show("닉네임: " + userName + "님 환영합니다!");
                return Tuple.Create(resultIdol, resultUser);
            }
            else if (isUserExist && !isIdolExist)                               //아이돌명이 없으면 
            {
                Idol newIdol = InsertNewIdolName(IdolName, nextCount("idol"));
                if (newIdol == null) return null;
                newIdol.IdolUsers.Add(resultUser.UName, 0);
                idolList.Add(newIdol);
                resultUser.UserIdols.Add(newIdol.IName, 0);
                MessageBox.Show("닉네임: " + userName + "님 환영합니다!");
                return Tuple.Create(newIdol, resultUser);
            }
            else if(!isUserExist && isIdolExist)
            {
                User newUser = InsertNewNickName(userName, nextCount("user"));
                if (newUser == null) return null;
                newUser.UserIdols.Add(resultIdol.IName, 0);
                userList.Add(newUser);
                resultIdol.IdolUsers.Add(newUser.UName, 0);
                return Tuple.Create(resultIdol, newUser);
            }
            else
            {
                Idol newIdol = InsertNewIdolName(IdolName, nextCount("idol"));
                User newUser = InsertNewNickName(userName, nextCount("user"));

                if (newIdol == null) return null;
                if (newUser == null) return null;
                newIdol.IdolUsers.Add(newUser.UName, 0);
                idolList.Add(newIdol);
                
                newUser.UserIdols.Add(newIdol.IName, 0);
                userList.Add(newUser);

                return Tuple.Create(newIdol, newUser);
            }
        }
        

        private User InsertNewNickName(string newNickName, int count)
        {
            MySqlConnection connection = new MySqlConnection(strconn);

            try
            {
                connection.Open();
                // 안전한 쿼리 구성을 위해 매개변수화된 쿼리 사용
                string sql = "INSERT INTO user(user_no, user_name) VALUES (@count, @newNickName)";
                MySqlCommand cmd = new MySqlCommand(sql, connection);

                // 매개변수 값 할당
                cmd.Parameters.AddWithValue("@count", count);
                cmd.Parameters.AddWithValue("@newNickName", newNickName);

                // INSERT 쿼리 실행
                cmd.ExecuteNonQuery();
                MessageBox.Show("사용자 정보 등록 완료");

                User newUser = new User();
                newUser.UName = newNickName;
                newUser.Unumber = count;

                return newUser;

            }
            catch (Exception ex)
            {
                MessageBox.Show("insertnewnickname Error 발생: " + ex.Message);
                return null;
            }
            finally
            {
                // 데이터베이스 연결 닫기
                if (connection.State == ConnectionState.Open)
                {
                    connection.Close();
                }
            }
        }


        private Idol InsertNewIdolName(string IdolName, int count)
        {
            MySqlConnection connection = new MySqlConnection(strconn);

            try
            {
                connection.Open();
                // 안전한 쿼리 구성을 위해 매개변수화된 쿼리 사용
                string sql = "INSERT INTO idol(idol_no, idol_name) VALUES (@count, @IdolName)";
                MySqlCommand cmd = new MySqlCommand(sql, connection);

                // 매개변수 값 할당
                cmd.Parameters.AddWithValue("@count", count);
                cmd.Parameters.AddWithValue("@IdolName", IdolName);

                // INSERT 쿼리 실행
                cmd.ExecuteNonQuery();
                MessageBox.Show("사용자 정보 등록 완료");

                Idol newIdol = new Idol(IdolName);
                newIdol.Inumber = count;
                return newIdol;
            }
            catch (Exception ex)
            {
                MessageBox.Show("insertnewidol() Error 발생: " + ex.Message);
                return null;
            }
            finally
            {
                // 데이터베이스 연결 닫기
                if (connection.State == ConnectionState.Open)
                {
                    connection.Close();
                }
            }

        }

        //점수추가
        //유저 번호와 아이돌 번호와 점수를 받아와 
        // 다 score 테이블에 저장

        public void updateScoreTable(int idolNum, int userNum, int point)
        {
            MySqlConnection connection = new MySqlConnection(strconn);

            try
            {
                connection.Open();
                // 안전한 쿼리 구성을 위해 매개변수화된 쿼리 사용
                string sql = "INSERT INTO score(user_no, idol_no, score_1) VALUES (@user_no, @idol_no, @point)";
                MySqlCommand cmd = new MySqlCommand(sql, connection);

                // 매개변수 값 할당
                cmd.Parameters.AddWithValue("@user_no", userNum);
                cmd.Parameters.AddWithValue("@idol_no", idolNum);
                cmd.Parameters.AddWithValue("@point", point);

                // INSERT 쿼리 실행
                cmd.ExecuteNonQuery();
                MessageBox.Show("사용자 정보 등록 완료");

            }
            catch (Exception ex)
            {
                MessageBox.Show("insertnewidol() Error 발생: " + ex.Message);
            }
            finally
            {
                // 데이터베이스 연결 닫기
                if (connection.State == ConnectionState.Open)
                {
                    connection.Close();
                }
            }
        }


        //새 객체를 만들어서 db에 집어넣을 때 호출
        public int nextCount(String tableName)
        {
            MySqlConnection connection = new MySqlConnection(strconn);
            string query = $"select count(*) as result from {tableName};" ;
            MySqlCommand cmd = new MySqlCommand(query, connection);

            int resultCount = 0;

            try
            {
                connection.Open();
                using (MySqlDataReader reader = cmd.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        resultCount = int.Parse(reader["result"].ToString());
                    }
                    reader.Close();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            return resultCount + 1;
        }


    }
}


